from typing import Any

import jax.tree_util as jtu

from .core import Config


def flatten_pytree(tree: Any, prefix: str = "", separator: str = ".") -> dict[str, Any]:
    """Flattens a JAX PyTree into a dictionary with dot-notation keys.

    This utility traverses the PyTree structure and produces a flat dictionary
    where keys represent the path to each leaf node using dot notation. This
    is particularly useful for logging complex nested `State` objects to
    flat metric logging systems like MLflow.

    Args:
        tree: The PyTree object to flatten (e.g., `State`, dict, list).
        prefix: An optional string to prepend to all generated keys.
        separator: The string used to separate path components. Defaults to ".".

    Returns:
        A dictionary mapping flattened string keys to the leaf values of the tree.

    Example:
        >>> tree = {"x": 10, "sub": {"y": 20}}
        >>> flatten_pytree(tree)
        {'x': 10, 'sub.y': 20}
    """
    flat_dict = {}

    # Get leaves with path
    # is_leaf=None means default JAX behavior (arrays are leaves, lists/tuples/dicts are nodes)
    leaves_with_path = jtu.tree_leaves_with_path(tree)

    for path, leaf in leaves_with_path:
        # Generate key from path
        # jax path entries are GetAttrKey, DictKey, SequenceKey, etc.
        key_parts = []
        for p in path:
            if isinstance(p, jtu.GetAttrKey):
                key_parts.append(p.name)
            elif isinstance(p, jtu.DictKey):
                key_parts.append(str(p.key))
            elif isinstance(p, jtu.SequenceKey):
                key_parts.append(str(p.idx))
            else:
                # Fallback
                key_parts.append(str(p))

        full_key = separator.join(key_parts)
        if prefix:
            full_key = f"{prefix}{separator}{full_key}"

        flat_dict[full_key] = leaf

    return flat_dict


def unflatten_pytree[T](flat_dict: dict[str, Any], structure: T, separator: str = ".") -> T:
    """Populates a PyTree structure with values from a flattened dictionary.

    This function attempts to reconstruct the values of a PyTree by looking up
    keys in a flattened dictionary that correspond to the paths in the `structure`.
    Leaves not found in the dictionary are kept as is from the structure.

    Args:
        flat_dict: A dictionary containing flattened keys and values.
        structure: A PyTree instance defining the target structure.
        separator: The separator used in the flattened keys. Defaults to ".".

    Returns:
        A new PyTree with the same structure as `structure`, but with leaves
        updated from `flat_dict`.

    Example:
        >>> flat = {"x": 100, "sub.y": 200}
        >>> structure = {"x": 0, "sub": {"y": 0}}
        >>> unflatten_pytree(flat, structure)
        {'x': 100, 'sub': {'y': 200}}
    """
    # We traverse the structure and look up values in flat_dict using the generated path key.

    def leaf_transform(path, leaf):
        key_parts = []
        for p in path:
            if isinstance(p, jtu.GetAttrKey):
                key_parts.append(p.name)
            elif isinstance(p, jtu.DictKey):
                key_parts.append(str(p.key))
            elif isinstance(p, jtu.SequenceKey):
                key_parts.append(str(p.idx))
            else:
                key_parts.append(str(p))

        key = separator.join(key_parts)

        if key in flat_dict:
            return flat_dict[key]
        return leaf  # Keep original leaf if not in dict (or raise error?)
        # For strict reconstruction, maybe we should warn?
        # But keeping original (default/placeholder) value is often desired for partial updates.

    return jtu.tree_map_with_path(leaf_transform, structure)


def flatten_config(config: Config, prefix: str = "", separator: str = ".") -> dict[str, Any]:
    """Flattens a Config object (Pydantic model) into a dictionary.

    It uses `model_dump()` to get the configuration as a dictionary and then
    flattens any nested dictionaries.

    Args:
        config: The `Config` instance to flatten.
        prefix: An optional string to prepend to keys.
        separator: The separator for nested keys. Defaults to ".".

    Returns:
        A flattened dictionary representation of the configuration.

    Example:
        >>> class MyConfig(Config):
        ...     nested: dict = {"a": 1}
        >>> config = MyConfig()
        >>> flatten_config(config)
        {'nested.a': 1}
    """
    # Use Pydantic's model_dump to get nested dict
    d = config.model_dump()

    # Flatten dict
    flat = {}

    def _recurse(curr, current_key):
        if isinstance(curr, dict):
            for k, v in curr.items():
                new_key = f"{current_key}{separator}{k}" if current_key else k
                _recurse(v, new_key)
        else:
            flat[current_key] = curr

    _recurse(d, prefix)
    return flat


def unflatten_config[C: Config](
    flat_dict: dict[str, Any], config_cls: type[C], separator: str = "."
) -> C:
    """Reconstructs a Config object from a flattened dictionary.

    This function unflattens a dictionary into a nested structure and then
    validates it against the Pydantic model `config_cls`.

    Args:
        flat_dict: The flattened dictionary containing configuration values.
        config_cls: The class of the `Config` object to reconstruct.
        separator: The separator used in the flattened keys. Defaults to ".".

    Returns:
        An instance of `config_cls` populated with values from `flat_dict`.

    Raises:
        ValueError: If there is a key conflict (e.g., 'a' is both a value and a container).
        ValidationError: If the reconstructed data fails Pydantic validation.

    Example:
        >>> flat = {"nested.a": 1}
        >>> unflatten_config(flat, MyConfig)
        MyConfig(nested={'a': 1})
    """
    # Unflatten dict
    nested = {}
    for key, value in flat_dict.items():
        parts = key.split(separator)
        curr = nested
        for _, part in enumerate(parts[:-1]):
            if part not in curr:
                curr[part] = {}
            curr = curr[part]
            if not isinstance(curr, dict):
                # Conflict: trying to use a value as a dict container
                # This might happen if keys are ambiguous e.g. "a" and "a.b"
                raise ValueError(f"Key conflict at '{part}' in '{key}'")

        curr[parts[-1]] = value

    return config_cls.model_validate(nested)
