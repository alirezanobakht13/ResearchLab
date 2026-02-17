from typing import Any, Type, TypeVar

import jax
import jax.tree_util as jtu
from pydantic import BaseModel

from .core import Config, State

T = TypeVar("T")
C = TypeVar("C", bound=Config)

def flatten_pytree(tree: Any, prefix: str = "", separator: str = ".") -> dict[str, Any]:
    """Flattens a PyTree (like State) into a dictionary with dot-notation keys.
    
    Args:
        tree: The PyTree to flatten.
        prefix: Optional prefix for keys.
        separator: Separator for nested keys.
        
    Returns:
        dict[str, Any]: Flattened dictionary.
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

def unflatten_pytree(flat_dict: dict[str, Any], structure: T, separator: str = ".") -> T:
    """Populates a PyTree structure with values from a flattened dictionary.
    
    Args:
        flat_dict: Dictionary with dot-notation keys.
        structure: A PyTree with the target structure.
        separator: Separator used in keys.
        
    Returns:
        T: The structure populated with values from flat_dict.
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
        return leaf # Keep original leaf if not in dict (or raise error?)
        # For strict reconstruction, maybe we should warn? 
        # But keeping original (default/placeholder) value is often desired for partial updates.
    
    return jtu.tree_map_with_path(leaf_transform, structure)

def flatten_config(config: Config, prefix: str = "", separator: str = ".") -> dict[str, Any]:
    """Flattens a Config object into a dictionary.
    
    Args:
        config: The Config object.
        prefix: Optional prefix.
        separator: Separator.
        
    Returns:
        dict[str, Any]: Flattened dict.
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

def unflatten_config(flat_dict: dict[str, Any], config_cls: Type[C], separator: str = ".") -> C:
    """Reconstructs a Config object from a flattened dictionary.
    
    Args:
        flat_dict: Flattened dictionary.
        config_cls: The Config class.
        separator: Separator.
        
    Returns:
        C: Instance of config_cls.
    """
    # Unflatten dict
    nested = {}
    for key, value in flat_dict.items():
        parts = key.split(separator)
        curr = nested
        for i, part in enumerate(parts[:-1]):
            if part not in curr:
                curr[part] = {}
            curr = curr[part]
            if not isinstance(curr, dict):
                 # Conflict: trying to use a value as a dict container
                 # This might happen if keys are ambiguous e.g. "a" and "a.b"
                 raise ValueError(f"Key conflict at '{part}' in '{key}'")
        
        curr[parts[-1]] = value
        
    return config_cls.model_validate(nested)
