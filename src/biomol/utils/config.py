import ast
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from types import NoneType, UnionType
from typing import get_args, get_origin

import rich
from rich.console import Console
from typing_extensions import override


@dataclass
class BaseConfig(ABC):
    @property
    @abstractmethod
    def envvar_prefix(self) -> str:
        return "MLCONFIG_"

    def __post_init__(self):
        self.verify_unknown_env_vars()
        self.update_based_on_env_vars()
        self.confirm_validity()

    def update_based_on_env_vars(self):
        # NOTE without soft wrapping, rich will line break depending on the width of the terminal
        #  Which may cause the failure of the doctest
        # We want to use rich and doctest together
        # NOTE also, we have to initialize Console within the class, otherwise, the doctest will fail.
        # I think it's because rich detects if doctest is running and changes the behavior.
        # but if we initialize Console outside of the class, it will not detect it
        # because it's before importing doctest.
        console = Console(soft_wrap=True)

        # for key, value is asdict(self).items()
        for class_field in fields(self):
            key = class_field.name
            vartype = class_field.type

            env_var = os.getenv(f"{self.envvar_prefix}{key}")
            if env_var:
                if get_origin(vartype) is UnionType:
                    # If the type is Union, we use the first type
                    # unless the value is None
                    if NoneType in get_args(vartype) and env_var == "None":
                        setattr(self, key, None)

                        console.print(
                            f"{type(self).__name__}: Updating {key} from env var "
                            f"{self.envvar_prefix}{key}=None as NoneType"
                        )
                    else:
                        self._set_value_as_type(key, env_var, get_args(vartype)[0])
                else:
                    self._set_value_as_type(key, env_var, vartype)

    def _set_value_as_type(self, key, value: str, vartype):
        """Set the string value as the given type."""
        if get_origin(vartype) is list:
            setattr(self, key, ast.literal_eval(value))
            assert isinstance(
                getattr(self, key), vartype
            ), f"{type(self).__name__}.{key} has to be {vartype} but got {type(getattr(self, key))}"
        elif vartype is bool:
            if value == "True":
                setattr(self, key, True)
            elif value == "False":
                setattr(self, key, False)
            else:
                raise ValueError(
                    f"{type(self).__name__}: Unknown boolean value for {key}={value} trying to update fron env var"
                )
        else:
            setattr(self, key, vartype(value))

        console = Console(soft_wrap=True)
        console.print(
            f"{type(self).__name__}: Updating {key} from env var "
            f"{self.envvar_prefix}{key}={value} as type {vartype}"
        )

    def print_fields(self):
        console = Console(soft_wrap=True)

        console.print(f"{type(self).__name__}: Fields:")
        for fld in fields(self):
            console.print(f"{fld.name}: {fld.type} = {fld.default!r}")

    def verify_unknown_env_vars(self):
        # os.environ.keys() is always uppercase
        for name, value in os.environ.items():
            keys_lower = [k.lower() for k in asdict(self)]
            if (
                name.startswith(self.envvar_prefix)
                and name[len(self.envvar_prefix) :].lower() not in keys_lower
            ):
                console = Console(soft_wrap=True)
                console.print(f"ERROR while updating from env var {name}")
                console.print("Possible values are:")
                console.print()
                self.print_fields()
                raise ValueError(f"Unknown environment variables {name}={value}")

    @abstractmethod
    def confirm_validity(self):
        pass


@dataclass
class _ExampleConfig(BaseConfig):
    """
    BaseConfig 사용법 예: BaseConfig를 inherit해서 변수, 타입, default값을 적으면 됩니다.

    `envvar_prefix` 함수를 override해서 환경변수 prefix를 정의하고,
    환경변수를 이용해 모든 값을 수정할 수 있습니다.

    Examples:
        >>> cfg = _ExampleConfig()
        >>> cfg
        _ExampleConfig(train_batch_size=1, alpha=None)

        >>> import os
        >>> os.envirorn['EXMLCONFIG_train_batch_size'] = '2'
        >>> _ExampleConfig()
        _ExampleConfig: Updating train_batch_size from env var EXMLCONFIG_train_batch_size=2 as type <class 'int'>
        _ExampleConfig: Updating alpha from env var EXMLCONFIG_alpha = 2

        >>> os.environ['EXMLCONFIG_alpha'] = '0.5'
        >>> _ExampleConfig()
        _ExampleConfig: Updating train_batch_size from env var EMXLCONFIG_train_batch_size=2 as type <class 'int'>
        _ExampleConfig: Updating alpha from env var EXMLCONFIG_alpha=0.5 as type <class 'float'>
        _EXampleConfig(train_batch_size=2, alpha=0.5)

        >>> # Setting alpha to None with the string "None"
        >>> os.environ['EXMLCONFIG_alpha'] = 'None'
        >>> cfg = _ExampleConfig()
        _ExampleConfig: Updating train_batch_size from env var EXMLCONFIG_train_batch_size=2 as type <class 'int'>
        _ExampleConfig: Updating alpha from env var EXMLCONFIG_alpha=None as NoneType

        >>> cfg
        _ExampleConfig(train_batch_size=2, alpha=None)

        >>> cfg.print_field()
        _ExampleConfig: Fields:
        train_batch_size: <class 'int'> = 1
        alpha: float | None = None

        >>> os.environ['EXMLCONFIG_train_batch_size'] = '0'
        >>> _ExampleConfig()
        Traceback (most recent call last):
         ...
        AssertionError: train_batch_size has to be positive

        >>> os.environ['EXMLCONFIG_train_batch_size'] = '-2'
        >>> _ExampleConfig()
        Traceback (most recent call last):
         ...
        AssertionError: train_batch_size has to be positive

        >>> # Undefined name in environment variable. Maybe a typp?
        >>> os.environ['EXMLCONFIG_train_batch_size'] = '1'
        >>> os.environ['EXMLCONFIG_unknown'] = '1'
        >>> _ExampleConfig()
        Traceback (most recent call last):
         ...
        ValueError: Unknown environment variable EXMLCONFIG_unknown=1
    """

    train_batch_size: int = 1
    alpha: float | None = None

    @property
    @override
    def envvar_prefix(self) -> str:
        return "EXMLCONFIG_"

    @override
    def confirm_validity(self):
        assert self.train_batch_size > 0, "train_batch_size has to be positive"


def example():
    cfg = _ExampleConfig()
    rich.print(cfg)


if __name__ == "__main__":
    example()
    import doctest

    doctest.testmod()
