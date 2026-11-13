# oj_setup.py
"""Настройка openJiuwen для запуска в контейнере (read-only FS, без лишнего шума в логах)."""
import os


def configure_openjiuwen_logging(level: str = "WARNING") -> None:
    """По умолчанию openJiuwen пишет в ./logs/ (в контейнере /app read-only) и логирует промпты целиком.

    Оставляем только консоль и уровень WARNING; файловый путь — во временный каталог.
    """
    from openjiuwen.core.common.logging.default.constant import DEFAULT_INNER_LOG_CONFIG
    from openjiuwen.core.common.logging.log_config import configure_log_config

    cfg = dict(DEFAULT_INNER_LOG_CONFIG)
    cfg.update({
        "level": level,
        "output": ["console"],
        "interface_output": ["console"],
        "performance_output": ["console"],
        "log_path": os.environ.get("OJ_LOG_DIR", "/tmp/ojlogs/"),
    })
    configure_log_config(cfg)