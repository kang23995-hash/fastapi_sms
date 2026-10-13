"""短信表的数据访问层。

所有 SQL 集中在这里，并且**一律使用 ``%s`` 占位符 + 参数元组**，
由驱动负责转义，从根本上杜绝 SQL 注入；任何位置都不做字符串拼接。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from database import Database, get_database
from models import SmsPage, SmsQueryFilter, SmsRecord

logger = logging.getLogger(__name__)

TABLE = "sms_records"

# 查询时显式列出字段，不用 SELECT *：字段顺序稳定，也不怕将来加字段把
# 结果集撑大（TEXT 类型的 message_content 在列表页是不需要的，
# 但列表页要展示正文，所以这里保留）。
_COLUMNS = (
    "id, message_unique_id, device_id, sim_slot, sim_phone_number, "
    "sender, sender_name, message_content, sms_vcode, receive_time, create_time"
)

# ---------------------------------------------------------------------
# 入库：靠 uk_message_unique_id 唯一索引实现去重
#
# ON DUPLICATE KEY UPDATE 后面跟一个「空更新」（把字段赋成它自己），
# 作用只是让重复数据不报错、也不修改原记录。此时 MySQL 的 affected_rows：
#     1 = 新插入成功
#     0 = 命中唯一索引、没有发生任何修改（即重复消息）
# 与 INSERT IGNORE 的区别：IGNORE 会把「字段超长」这类真正的错误也一起吞掉，
# 这里要的是「只忽略重复」，所以用 ON DUPLICATE KEY。
# ---------------------------------------------------------------------
_INSERT_SQL = f"""
INSERT INTO {TABLE}
    (message_unique_id, device_id, sim_slot, sim_phone_number,
     sender, sender_name, message_content, sms_vcode, receive_time)
VALUES
    (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE message_unique_id = message_unique_id
"""


class SmsRepository:
    """短信记录的数据访问对象。"""

    def __init__(self, db: Optional[Database] = None) -> None:
        self._db = db or get_database()

    # ------------------------------------------------------------ 写入

    async def insert_sms(
        self,
        *,
        message_unique_id: str,
        device_id: str,
        sim_slot: int,
        sim_phone_number: str,
        sender: str,
        sender_name: str,
        message_content: str,
        sms_vcode: str,
        receive_time: Any,
    ) -> tuple[bool, Optional[int]]:
        """入库一条短信。

        整个动作在事务里完成（见 :meth:`Database.run_in_transaction`），
        异常时自动回滚。``sender_name`` 与 ``sms_vcode`` 由上层从正文里
        解析出来后一并传入。

        :return: ``(是否新插入, 新记录的主键 id)``；
                 命中唯一索引（重复消息）时返回 ``(False, None)``，
                 并且**不会修改已有记录**。
        """

        args = (
            message_unique_id,
            device_id,
            sim_slot,
            sim_phone_number,
            sender,
            sender_name,
            message_content,
            sms_vcode,
            receive_time,
        )

        async def operation(cursor) -> tuple[bool, Optional[int]]:
            await cursor.execute(_INSERT_SQL, args)
            affected = cursor.rowcount
            if affected == 1:
                # 新插入，lastrowid 就是自增主键
                return True, cursor.lastrowid
            # 0（重复，无变化）或 2（重复且发生了更新）都按「已存在」处理
            return False, None

        return await self._db.run_in_transaction(operation)

    # ------------------------------------------------------------ 查询

    async def query_page(
        self,
        filters: SmsQueryFilter,
        page: int,
        page_size: int,
    ) -> SmsPage:
        """分页查询，按短信接收时间倒序（最新在最前）。"""
        where_sql, where_args = self._build_where(filters)

        # 总数与设备数一次查出，省一次数据库往返
        count_sql = (
            f"SELECT COUNT(*) AS total, COUNT(DISTINCT device_id) AS devices "
            f"FROM {TABLE} {where_sql}"
        )
        count_row = await self._db.fetch_one(count_sql, where_args) or {}
        total = int(count_row.get("total") or 0)

        offset = max(0, (page - 1) * page_size)

        # ORDER BY receive_time DESC, id DESC：
        #   * receive_time 上有索引，倒序取前 N 条可以直接走索引
        #   * 再按 id 倒序是为了同一秒内的多条短信顺序稳定
        # LIMIT / OFFSET 同样用参数占位符
        list_sql = (
            f"SELECT {_COLUMNS} FROM {TABLE} {where_sql} "
            f"ORDER BY receive_time DESC, id DESC LIMIT %s OFFSET %s"
        )
        rows = await self._db.fetch_all(list_sql, (*where_args, page_size, offset))

        return SmsPage(
            items=[SmsRecord.from_row(row) for row in rows],
            total=total,
            page=page,
            page_size=page_size,
        )

    async def count_devices(self) -> int:
        """在线设备数（历史出现过的设备数），页面顶部展示用。"""
        value = await self._db.fetch_value(
            f"SELECT COUNT(DISTINCT device_id) AS c FROM {TABLE}"
        )
        return int(value or 0)

    # ------------------------------------------------------------ 删除

    async def delete_by_ids(self, ids: list[int]) -> int:
        """按主键 id 批量删除（单选时传一个 id 即可），返回删除条数。

        ``IN (%s, %s, ...)`` 的占位符个数按列表长度动态生成，
        **值仍全部走参数元组**，不做字符串拼接，杜绝注入。
        """
        if not ids:
            return 0
        placeholders = ", ".join(["%s"] * len(ids))
        sql = f"DELETE FROM {TABLE} WHERE id IN ({placeholders})"
        return await self._db.execute(sql, tuple(ids))

    async def delete_all(self) -> int:
        """清空整张表（不可逆），返回删除条数。

        这里用 ``DELETE`` 而不是 ``TRUNCATE``：DELETE 走事务、能被
        :meth:`Database.run_in_transaction` 的异常回滚保护，语义更安全；
        TRUNCATE 是 DDL，无法回滚且会重置自增 id。
        """
        return await self._db.execute(f"DELETE FROM {TABLE}")

    # ------------------------------------------------------------ 内部

    @staticmethod
    def _build_where(filters: SmsQueryFilter) -> tuple[str, tuple[Any, ...]]:
        """把筛选条件拼成 ``WHERE`` 子句。

        仍然全部使用占位符：值通过参数元组传入，条件本身只由代码里的
        固定字符串拼成，用户输入永远不会成为 SQL 的一部分。

        注意：这里是「包含匹配」（``%关键字%``），对排障最友好，但前置
        通配符会让索引失效、退化成扫描。数据量大之后如果筛选明显变慢，
        可把下面三处改成 ``%s`` + ``f"{value}%"``（前缀匹配），
        即可命中 idx_device_id / idx_sender / idx_sim_phone。
        """
        conditions: list[str] = []
        args: list[Any] = []

        if filters.device_id:
            conditions.append("device_id LIKE %s")
            args.append(f"%{filters.device_id}%")
        if filters.sim_phone_number:
            conditions.append("sim_phone_number LIKE %s")
            args.append(f"%{filters.sim_phone_number}%")
        if filters.sender:
            conditions.append("sender LIKE %s")
            args.append(f"%{filters.sender}%")

        where_sql = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        return where_sql, tuple(args)


_repository: Optional[SmsRepository] = None


def get_repository() -> SmsRepository:
    """获取进程内唯一的数据访问对象。"""
    global _repository
    if _repository is None:
        _repository = SmsRepository()
    return _repository
