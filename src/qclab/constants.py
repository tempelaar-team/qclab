"""
This module contains the Constants class.
"""


class Constants:
    """
    Constants class to define constants and trigger updates when they change.
    """

    def __init__(self, update_function=None):
        self._updating = False
        self._init_complete = False
        self._update_function = update_function

    def __setattr__(self, name, value):
        """
        Overrides attribute setting to call the update function after the attribute is
        changed, preventing recursion.
        """
        super().__setattr__(name, value)
        if not self._updating and name not in {
            "_updating",
            "_update_function",
            "_init_complete",
        }:
            if self._init_complete:
                self._updating = True
                if self._update_function is not None:
                    self._update_function()
                self._updating = False

    def get(self, name, default=None):
        """
        Get the value of an attribute, returning default if it does not exist.

        .. rubric:: Args




        name : str
            The name of the attribute to get.
        default : any, optional
            The default value to return if the attribute does not exist.

        .. rubric:: Returns

        self.name : any
            The value of the attribute.
        """
        return getattr(self, name, default)
