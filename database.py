"""MySQL 异步连接池与事务封装。

设计要点
--------
* **连接池**：用 aiomysql 连接池，连接在进程内复用。绝不每个请求
  新建/销毁连接 —— 那是压测时最先被打爆的地方（TCP 握手 + MySQL 认证
  的开销远大于 SQL 本身）。
* **事务**：所有读写都在显式事务里执行，异常自动 ``ROLLBACK``，
  保证「要么整条入库，要么什么都没发生」，避免出现半截数据。
* **断连自愈**：对 ``MySQL server has gone away``、``Lost connection``
  这类可恢复错误，自动丢弃坏连接、重建连接池并重试；
  数据库中途重启也不用重启本服务。
* **超时保护**：每条 SQL 都套了 ``asyncio.wait_for``，慢查询会快速失败，
  不会把 worker 进程拖死。
* **参数化查询**：所有 SQL 一律用 ``%s`` 占位符 + 参数元组，
  由驱动做转义，杜绝 SQL 注入。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional, TypeVar

import aiomysql

from config import Settings, get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 连接层面的可恢复错误码，出现这些才值得重试
#   2006 MySQL server has gone away
#   2013 Lost connection to MySQL server during query
#   2055 Lost connection to MySQL server at 'reading authorization packet'
#   1927 Connection was killed
#   1047 Unknown command（连接状态错乱）
#   1053 Server shutdown in progress
_RETRYABLE_ERRNOS = frozenset({2006, 2013, 2055, 1927, 1047, 1053})


class DatabaseUnavailableError(RuntimeError):
    """数据库不可用（连不上、连接池未就绪、SQL 超时）。

    上层捕获后应返回 503，让 APP 按重试策略继续重试，
    而不是返回 500 让 APP 以为报文本身有问题。
    """


class Database:
    """连接池 + 事务的封装，进程内单例使用。"""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._pool: Optional[aiomysql.Pool] = None
        # 保护连接池的创建/重建，避免并发请求同时重建出一堆池子
        self._pool_lock = asyncio.Lock()

    # ------------------------------------------------------------ 生命周期

    async def connect(self) -> bool:
        """启动时建立连接池。失败只记日志不抛异常。

        这样即使 MySQL 暂时没起来，服务仍然能启动：``GET /health`` 会
        如实报告不可用，页面会给出提示，等 MySQL 恢复后连接池会自动重建。
        """
        try:
            await self._get_pool()
            return True
        except DatabaseUnavailableError as exc:
            logger.critical(
                "启动时无法连接 MySQL（%s:%s/%s）：%s。服务将继续启动，"
                "数据库恢复后会自动重连。",
                self._settings.db_host,
                self._settings.db_port,
                self._settings.db_name,
                exc,
            )
            return False

    async def close(self) -> None:
        """优雅关闭：释放所有连接。"""
        async with self._pool_lock:
            await self._close_pool_locked()

    async def is_healthy(self) -> tuple[bool, str]:
        """健康检查：真正跑一次 SELECT 1，而不是只看池子是否存在。"""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("SELECT 1")
                    await cursor.fetchone()
            return True, "ok"
        except Exception as exc:  # noqa: BLE001 —— 健康检查要吞掉所有异常
            return False, f"{type(exc).__name__}: {exc}"

    # ------------------------------------------------------------ 对外执行接口

    async def run_in_transaction(
        self,
        operation: Callable[[aiomysql.cursors.DictCursor], Awaitable[T]],
        *,
        timeout: Optional[float] = None,
        retry: bool = True,
    ) -> T:
        """在事务中执行 ``operation(cursor)``。

        * 正常结束 -> ``COMMIT``
        * 抛异常   -> ``ROLLBACK`` 后原样抛出
        * 连接类异常 -> 重建连接池并重试（最多 ``DB_RETRY_TIMES`` 次）
        * 超时     -> 直接抛 :class:`DatabaseUnavailableError`（不重试，
          避免雪上加霜）
        """
        query_timeout = timeout or self._settings.db_query_timeout
        max_attempts = self._settings.db_retry_times if retry else 1

        last_error: Optional[BaseException] = None

        for attempt in range(1, max_attempts + 1):
            try:
                return await asyncio.wait_for(
                    self._execute_once(operation), timeout=query_timeout
                )
            except asyncio.TimeoutError as exc:
                logger.error(
                    "SQL 执行超时（>%.1fs），本次请求判定为失败", query_timeout
                )
                raise DatabaseUnavailableError(
                    f"数据库响应超时（超过 {query_timeout:g} 秒）"
                ) from exc
            except DatabaseUnavailableError:
                # 池子没建起来，不属于「重试能解决」的范畴，直接上抛
                raise
            except (aiomysql.OperationalError, aiomysql.InterfaceError) as exc:
                last_error = exc
                if not self._is_retryable(exc) or attempt >= max_attempts:
                    logger.error(
                        "数据库操作失败（第 %d/%d 次尝试，不可重试或已用尽重试）: %s",
                        attempt,
                        max_attempts,
                        exc,
                    )
                    raise DatabaseUnavailableError(str(exc)) from exc

                logger.warning(
                    "数据库连接异常，第 %d/%d 次尝试失败，正在重建连接池后重试: %s",
                    attempt,
                    max_attempts,
                    exc,
                )
                await self._recycle_pool()
                # 退避，给数据库一点恢复时间
                await asyncio.sleep(min(0.2 * attempt, 1.0))

        # 理论上到不了这里
        raise DatabaseUnavailableError(str(last_error))

    # ------------------------------------------------------------ 读操作便捷方法

    async def fetch_all(
        self, sql: str, args: Optional[tuple[Any, ...]] = None
    ) -> list[dict[str, Any]]:
        """查询多行，返回 ``[{列名: 值}, ...]``。"""

        async def operation(cursor: aiomysql.cursors.DictCursor) -> list[dict[str, Any]]:
            await cursor.execute(sql, args or ())
            return list(await cursor.fetchall())

        return await self.run_in_transaction(operation)

    async def fetch_one(
        self, sql: str, args: Optional[tuple[Any, ...]] = None
    ) -> Optional[dict[str, Any]]:
        """查询单行，无结果返回 ``None``。"""

        async def operation(cursor: aiomysql.cursors.DictCursor) -> Optional[dict[str, Any]]:
            await cursor.execute(sql, args or ())
            return await cursor.fetchone()

        return await self.run_in_transaction(operation)

    async def fetch_value(
        self, sql: str, args: Optional[tuple[Any, ...]] = None
    ) -> Any:
        """查询单个标量值（取第一行第一列）。"""

        row = await self.fetch_one(sql, args)
        if not row:
            return None
        return next(iter(row.values()))

    async def execute(
        self, sql: str, args: Optional[tuple[Any, ...]] = None
    ) -> int:
        """执行单条写语句，返回受影响行数。

        需要多条语句组成一个原子操作时，请使用 :meth:`run_in_transaction`，
        把多条语句放进同一个 ``operation`` 回调里。
        """

        async def operation(cursor: aiomysql.cursors.DictCursor) -> int:
            await cursor.execute(sql, args or ())
            return cursor.rowcount

        return await self.run_in_transaction(operation)

    # ------------------------------------------------------------ 内部实现

    async def _execute_once(
        self, operation: Callable[[aiomysql.cursors.DictCursor], Awaitable[T]]
    ) -> T:
        """一次「取连接 -> 开事务 -> 执行 -> 提交/回滚」的完整动作。"""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            # DictCursor：查询结果直接是 dict，配合模板渲染更省事
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await conn.begin()
                try:
                    result = await operation(cursor)
                except Exception:
                    # 任何异常都回滚，保证原子性
                    await conn.rollback()
                    raise
                else:
                    await conn.commit()
                    return result

    async def _get_pool(self) -> aiomysql.Pool:
        """取连接池；不存在（或刚被回收）时惰性重建。"""
        if self._pool is not None:
            return self._pool

        async with self._pool_lock:
            # 双重检查：可能已被其它协程建好
            if self._pool is None:
                await self._create_pool_locked()
            assert self._pool is not None
            return self._pool

    async def _create_pool_locked(self) -> None:
        """在持有锁的前提下建立连接池。"""
        s = self._settings
        try:
            self._pool = await aiomysql.create_pool(
                host=s.db_host,
                port=s.db_port,
                user=s.db_user,
                password=s.db_password,
                db=s.db_name,
                # utf8mb4 才能正确存取 emoji / 生僻字
                charset="utf8mb4",
                # 关掉自动提交，事务由我们显式控制
                autocommit=False,
                minsize=max(1, s.db_pool_min),
                maxsize=max(1, s.db_pool_max),
                connect_timeout=s.db_connect_timeout,
                echo=False,
            )
            logger.info(
                "MySQL 连接池已建立：%s:%s/%s（连接数 %d~%d）",
                s.db_host,
                s.db_port,
                s.db_name,
                s.db_pool_min,
                s.db_pool_max,
            )
        except Exception as exc:
            self._pool = None
            logger.critical(
                "创建 MySQL 连接池失败：%s（请检查 .env 的 DB_* 配置、MySQL 是否启动、"
                "账号密码是否正确）",
                exc,
                exc_info=True,
            )
            raise DatabaseUnavailableError(f"无法连接数据库：{exc}") from exc

    async def _recycle_pool(self) -> None:
        """丢弃当前连接池并重建（断连自愈的核心）。"""
        async with self._pool_lock:
            await self._close_pool_locked()
            try:
                await self._create_pool_locked()
            except DatabaseUnavailableError:
                # 重建也失败就留给下一次请求再试，不在这里继续纠缠
                logger.error("重建连接池失败，等待下一个请求触发重连")

    async def _close_pool_locked(self) -> None:
        if self._pool is None:
            return
        pool = self._pool
        self._pool = None
        try:
            pool.close()
            await pool.wait_closed()
        except Exception as exc:  # noqa: BLE001
            logger.warning("关闭连接池时出现异常（可忽略）: %s", exc)

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        """判断是否属于「连接坏了，重试有意义」的错误。"""
        if not exc.args:
            return False
        first = exc.args[0]
        return isinstance(first, int) and first in _RETRYABLE_ERRNOS


# ---------------------------------------------------------------- 单例
_database: Optional[Database] = None


def get_database() -> Database:
    """获取进程内唯一的 Database 实例。"""
    global _database
    if _database is None:
        _database = Database()
    return _database
