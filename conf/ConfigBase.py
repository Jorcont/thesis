from dataclasses import dataclass
from typing import TypeVar, Type, Optional, ClassVar

T = TypeVar('T', bound='ConfigBase')


class ConfigBase:
    _instance: ClassVar[Optional['ConfigBase']] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        original_post_init = getattr(cls, '__post_init__', None)

        def new_post_init(self):
            type(self)._instance = self
            if original_post_init:
                original_post_init(self)

        cls.__post_init__ = new_post_init

    @classmethod
    def get_config(cls: Type[T]) -> T | None:
        return cls._instance  # type: ignore

    @classmethod
    def to_dict(cls):
        from dataclasses import asdict

        if not hasattr(cls._instance, "config") or cls._instance.config is None:
            return {}

        try:
            return asdict(cls._instance.config)
        except TypeError:
            return dict(getattr(cls._instance.config, "__dict__", {}))
