# SPDX-License-Identifier: Apache-2.0

from typing import Iterable

import numpy as np


class StructuredVector:
    """A flat numpy vector composed of named, contiguous segments.

    Segments are appended one at a time with `add_slice`. Each segment is
    accessible by name via dict-style indexing and returns a view into the
    underlying `vec` buffer, so writes propagate.

    Typical use is to declare the layout once at construction time, then read
    and write segments by name during the program's hot loop.

    Attributes:
        vec: The underlying 1-D numpy array holding all segment data.

    Example:
        Build incrementally and access by name::

            obs = (StructuredVector()
                   .add_slice("q",   13)
                   .add_slice("v",   12)
                   .add_slice("tau", 12, values=initial_tau))

            obs["q"]               # np.ndarray view of length 13
            obs["tau"] = new_tau   # in-place write into obs.vec
            np.asarray(obs)        # full underlying vector

        Reuse a segment's slice on a parallel array (e.g. a Jacobian column)::

            J[:, obs.slice_of("v")]  # columns corresponding to velocities

    Note:
        Each call to `add_slice` reallocates `vec` via `np.concatenate`, which
        invalidates any previously-held view. Finish building the layout
        before caching views or passing `obs.vec` to other code.
    """

    def __init__(self):
        """Initializes an empty vector with no registered segments."""
        self.vec = np.zeros(0)
        self._slices: dict[str, slice] = {}

    @property
    def dim(self) -> int:
        """int: Total length of the underlying vector."""
        return self.vec.size

    def add_slice(
        self,
        name: str,
        length: int,
        values: np.ndarray | float = 0.0,
    ) -> "StructuredVector":
        """Appends a new named segment to the vector.

        Args:
            name: Identifier for the segment. Must be unique within this
                vector.
            length: Number of elements in the segment.
            values: Initial contents. A scalar fills the segment uniformly;
                an array must have shape `(length,)`. Defaults to `0.0`.

        Returns:
            `self`, to enable method chaining.

        Raises:
            KeyError: If `name` is already registered.
            ValueError: If `values` is an array of the wrong length.

        Example:
            >>> sv = StructuredVector().add_slice("q", 3)
            >>> sv.add_slice("v", 2, values=1.0)
            >>> sv["v"]
            array([1., 1.])

        Note:
            Reallocates `self.vec`; previously-held views become stale.
        """
        if name in self._slices:
            raise KeyError(f"slice {name!r} already exists")
        start, stop = self.vec.size, self.vec.size + length
        self._slices[name] = slice(start, stop)
        chunk = (
            np.full(length, values, dtype=float)
            if np.ndim(values) == 0
            else np.asarray(values, dtype=float).reshape(length)
        )
        self.vec = np.concatenate([self.vec, chunk])
        return self

    def __getitem__(self, name: str) -> np.ndarray:
        """Returns a view into the segment named `name`.

        Writes to the returned array mutate `self.vec` in place.

        Args:
            name: Segment name.

        Returns:
            A 1-D view into the underlying buffer.

        Raises:
            KeyError: If `name` is not a registered segment.

        Example:
            >>> sv["q"][0] = 1.5   # mutates sv.vec[0]
        """
        return self.vec[self._slices[name]]

    def __setitem__(self, name: str, value) -> None:
        """Overwrites the segment named `name`.

        Args:
            name: Segment name.
            value: Replacement data; must be broadcastable to the segment's
                length.

        Raises:
            KeyError: If `name` is not a registered segment.

        Example:
            >>> sv["tau"] = np.zeros(12)
            >>> sv["tau"] = 0.0          # broadcasts to the full segment
        """
        self.vec[self._slices[name]] = value

    def __contains__(self, name: str) -> bool:
        """Returns whether `name` is a registered segment.

        Args:
            name: Segment name.

        Example:
            >>> "q" in sv
            True
        """
        return name in self._slices

    def names(self) -> Iterable[str]:
        """Returns the segment names in insertion order.

        Returns:
            A view over the registered segment names.

        Example:
            >>> list(sv.names())
            ['q', 'v', 'tau']
        """
        return self._slices.keys()

    def slice_of(self, name: str) -> slice:
        """Returns the `slice` object for the named segment.

        Useful for indexing parallel arrays that share this layout, such as a
        target vector, a Jacobian, or a covariance matrix.

        Args:
            name: Segment name.

        Returns:
            The slice object spanning the segment in `vec`.

        Raises:
            KeyError: If `name` is not a registered segment.

        Example:
            >>> jacobian[:, sv.slice_of("v")]   # columns for velocity block
            >>> target[sv.slice_of("q")] = q_des
        """
        return self._slices[name]

    def __array__(self, dtype=None) -> np.ndarray:
        """Numpy protocol hook returning the full underlying vector.

        Lets `np.asarray(sv)` work transparently.

        Args:
            dtype: Optional dtype; if provided, returns a copy cast to it.

        Returns:
            The full 1-D vector (a view when `dtype` is `None`).

        Example:
            >>> np.asarray(sv).shape
            (37,)
        """
        return self.vec if dtype is None else self.vec.astype(dtype)

    def __repr__(self) -> str:
        """Returns a human-readable representation listing each segment."""
        parts = ", ".join(f"{n}={self[n]}" for n in self._slices)
        return f"StructuredVector({parts})"
