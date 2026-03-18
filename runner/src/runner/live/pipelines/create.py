"""@pipeline decorator for registering live pipelines.

The decorator handles two forms:
- **Function form**: wraps a plain function into a Pipeline subclass.
- **Class form**: registers an existing Pipeline subclass (or plain class
  with a ``transform`` method) and attaches a ``PipelineSpec``.

Lifecycle hooks (defined on the Pipeline base class):
* ``prepare_models``: called at build time (model download, TensorRT compile)
* ``on_ready``: called once at startup
* ``transform``: called per frame
* ``on_update``: called when params change mid-stream
* ``on_stop``: called on shutdown
"""

import logging
from typing import Optional, Type

from .interface import Pipeline, BaseParams
from .spec import PipelineSpec


def pipeline(
    name: str,
    params: Optional[Type[BaseParams]] = None,
    initial_params: Optional[dict] = None,
):
    """Decorator to define a pipeline. Can decorate a function or class.

    Args:
        name: Pipeline identifier. Must match the MODEL_ID on the orchestrator.
        params: Pydantic model for pipeline parameters. Defaults to BaseParams.
        initial_params: Default parameter values passed on init.
    """
    params_cls = params or BaseParams

    def decorator(func_or_class):
        if callable(func_or_class) and not isinstance(func_or_class, type):
            # Function form: wrap into a Pipeline subclass with a transform method.
            func = func_or_class

            class GeneratedPipeline(Pipeline):
                def transform(self, frame, p):
                    # Note: _invoke is called by the framework, not here.
                    # For function form we store the original func for the
                    # framework to call directly.
                    raise NotImplementedError(
                        "Function-form pipelines are called via _original_func"
                    )

            # Store original function for the framework to call.
            GeneratedPipeline._original_func = staticmethod(func)
            GeneratedPipeline._is_function_form = True
            GeneratedPipeline.__name__ = func.__name__
            GeneratedPipeline.__qualname__ = func.__qualname__
            GeneratedPipeline.__module__ = func.__module__
            pipeline_cls = GeneratedPipeline
        elif isinstance(func_or_class, type):
            user_cls = func_or_class

            if not hasattr(user_cls, "transform"):
                raise TypeError(
                    f"@pipeline class {user_cls.__name__} must define a "
                    f"'transform' method"
                )

            # If not already a Pipeline subclass, wrap it into one.
            if not issubclass(user_cls, Pipeline):
                pipeline_cls = _wrap_plain_class(user_cls)
            else:
                pipeline_cls = user_cls
                pipeline_cls._is_function_form = False
        else:
            raise TypeError(
                "@pipeline can only decorate a function or class, got "
                f"{type(func_or_class)}"
            )

        if "." in pipeline_cls.__qualname__:
            logging.warning(
                f"@pipeline decorating nested '{pipeline_cls.__qualname__}' — "
                f"this may cause import issues if the enclosing scope is not "
                f"accessible"
            )

        # Attach params class for the framework to use.
        pipeline_cls._params_cls = params_cls

        # Auto-generate PipelineSpec.
        params_import = (
            f"{params_cls.__module__}:{params_cls.__qualname__}"
            if params_cls is not BaseParams
            else None
        )
        pipeline_cls._spec = PipelineSpec(
            name=name,
            pipeline_cls=f"{pipeline_cls.__module__}:{pipeline_cls.__qualname__}",
            params_cls=params_import,
            initial_params=initial_params or {},
        )

        return pipeline_cls

    return decorator


def _wrap_plain_class(user_cls) -> Type[Pipeline]:
    """Wrap a plain class (not a Pipeline subclass) into a Pipeline subclass.

    This supports backward compatibility with plain classes that define
    transform/on_ready/on_update/on_stop/prepare_models by name.
    """

    class GeneratedPipeline(Pipeline):
        def __init__(self):
            self._inner = user_cls()

        def transform(self, frame, params):
            return self._inner.transform(frame, params)

    # Forward lifecycle hooks if defined on the user class.
    if hasattr(user_cls, "on_ready"):
        GeneratedPipeline.on_ready = lambda self, **params: self._inner.on_ready(**params)

    if hasattr(user_cls, "on_update"):
        GeneratedPipeline.on_update = lambda self, **params: self._inner.on_update(**params)

    if hasattr(user_cls, "on_stop"):
        GeneratedPipeline.on_stop = lambda self: self._inner.on_stop()

    if hasattr(user_cls, "prepare_models"):
        GeneratedPipeline.prepare_models = classmethod(
            lambda cls: user_cls.prepare_models()
        )

    GeneratedPipeline._is_function_form = False
    GeneratedPipeline.__name__ = user_cls.__name__
    GeneratedPipeline.__qualname__ = user_cls.__qualname__
    GeneratedPipeline.__module__ = user_cls.__module__

    return GeneratedPipeline
