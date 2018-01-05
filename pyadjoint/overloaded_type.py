from .tape import get_working_tape, annotate_tape
from .block_variable import BlockVariable


class OverloadedType(object):
    """Base class for OverloadedType types.

    The purpose of each OverloadedType is to extend a type such that
    it can be referenced by blocks as well as overload basic mathematical
    operations such as __mul__, __add__, where they are needed.

    Abstract methods:
        :func:`adj_update_value`

    """
    def __init__(self, *args, **kwargs):
        self.original_block_variable = self.create_block_variable()

    def create_block_variable(self):
        self.block_variable = BlockVariable(self)
        return self.block_variable

    @property
    def adj_value(self):
        return self.original_block_variable.adj_value

    @adj_value.setter
    def adj_value(self, value):
        self.block_variable.adj_value = value

    @property
    def tlm_value(self):
        return self.original_block_variable.tlm_value

    @tlm_value.setter
    def tlm_value(self, value):
        self.original_block_variable.tlm_value = value

    def _ad_convert_type(self, value, options={}):
        """This method must be overridden.

        Should implement a way to convert the result of an adjoint computation, `value`,
        into the same type as `self`.

        Args:
            value (Any): The value to convert. Should be a result of an adjoint computation.
            options (dict): A dictionary with options that may be supplied by the user.
                If the convert type functionality offers some options on how to convert,
                this is the dictionary that should be used.
                For an example see fenics_adjoint.types.Function

        Returns:
            OverloadedType: An instance of the same type as `self`.

        """
        raise NotImplementedError

    def _ad_create_checkpoint(self):
        """This method must be overridden.
        
        Should implement a way to create a checkpoint for the overloaded object.
        The checkpoint should be returned and possible to restore from in the
        corresponding _ad_restore_at_checkpoint method.

        Returns:
            :obj:`object`: A checkpoint. Could be of any type, but must be possible
                to restore an object from that point.

        """
        raise NotImplementedError

    def _ad_restore_at_checkpoint(self, checkpoint):
        """This method must be overridden.

        Should implement a way to restore the object at supplied checkpoint.
        The checkpoint is created from the _ad_create_checkpoint method.

        Returns:
            :obj:`OverloadedType`: The object with same state as at the supplied checkpoint.

        """
        raise NotImplementedError

    def adj_update_value(self, value):
        """This method must be overridden.

        The method should implement a routine for assigning a new value
        to the overloaded object.

        Args:
            value (:obj:`object`): Should be an instance of the OverloadedType.

        """
        raise NotImplementedError

    def _ad_mul(self, other):
        """This method must be overridden.

        The method should implement a routine for multiplying the overloaded object
        with another object, and return an object of the same type as `self`.

        Args:
            other (:obj:`object`): The object to be multiplied with this.
                Should at the very least accept :obj:`float` and :obj:`integer` objects.

        Returns:
            :obj:`OverloadedType`: The product of the two objects represented as
                an instance of the same subclass of :class:`OverloadedType` as the type
                of `self`.

        """
        raise NotImplementedError

    def _ad_add(self, other):
        """This method must be overridden.

        The method should implement a routine for adding the overloaded object
        with another object, and return an object of the same type as `self`.

        Args:
            other (:obj:`object`): The object to be added with this.
                Should at the very least accept objects of the same type as `self`.

        Returns:
            :obj:`OverloadedType`: The sum of the two objects represented as
                an instance of the same subclass of :class:`OverloadedType` as the type
                of `self`.

        """
        raise NotImplementedError

    def _ad_dot(self, other):
        """This method must be overridden.

        The method should implement a routine for computing the dot product of
        the overloaded object with another object of the same type, and return
        a :obj:`float`.

        Args:
            other (:obj:`OverloadedType`): The object to compute the dot product with.
                Should be of the same type as `self`.

        Returns:
            :obj:`float`: The dot product of the two objects.

        """
        raise NotImplementedError

    def _ad_will_add_as_dependency(self):
        """Method called when the object is added as a Block dependency.

        Returns:
            bool: True if the saved checkpoint should be overwritten.

        """
        return False

    def _ad_will_add_as_output(self):
        """Method called when the object is added as a Block output.

        Returns:
            bool: True if the saved checkpoint should be overwritten.

        """
        return True

    @staticmethod
    def _ad_assign_numpy(dst, src, offset):
        """This method must be overridden.

        The method should implement a routine for assigning the values from
        a numpy array `src` to the checkpoint `dst`. `dst` should be an instance
        of the implementing class.

        Args:
            dst (obj): The object which should be assigned new values.
                The type will most likely be an OverloadedType or similar.
            src (numpy.ndarray): The numpy array to use as a source for the assignment.
                `src` should have the same underlying dimensions as `dst`.
            offset:

        Returns:
            obj: The `dst` object. If `dst` is mutable it is preferred to be the same
                instance as supplied to the function call. Otherwise a new instance
                must be initialized and returned with the correct `src` values.
            int: The new offset.

        """
        raise NotImplementedError

    @staticmethod
    def _ad_to_list(m):
        """This method must be overridden.

        The method should implement a routine for converting `m` into a
        list type. `m` should be an instance of the same type as the class
        this is method is implemented in. Although maybe the backend version
        of this class, meaning it is not necessarily an OverloadedType.

        Args:
            m (obj): The object to be converted into a list.

        Returns:
            list: A list representation of the data structure of `m`.

        """
        raise NotImplementedError

    def _ad_copy(self):
        """This method must be overridden.

        The method should implement a routine for copying itself.

        Returns:
            OverloadedType: A (deep) copy of `self`.

        """
        raise NotImplementedError

    def _ad_dim(self):
        """This method must be overridden.

        The method should implement a routine for computing the number of components
        of `self`.

        Returns:
            int: The number of components of `self`.

        """
        raise NotImplementedError


class FloatingType(OverloadedType):
    def __init__(self, *args, **kwargs):
        self.block_class = kwargs.pop("block_class", None)
        self._ad_args = kwargs.pop("_ad_args", [])
        self._ad_kwargs = kwargs.pop("_ad_kwargs", {})
        self._ad_floating_active = kwargs.pop("_ad_floating_active", False)
        self.annotate_tape = annotate_tape(kwargs)
        self.block = None

        self.output_block = None
        self._ad_output_args = kwargs.pop("_ad_output_args", [])
        self._ad_output_kwargs = kwargs.pop("_ad_output_kwargs", {})
        self.output_block_class = kwargs.pop("output_block_class", None)
        self._ad_outputs = kwargs.pop("_ad_outputs", [])
        OverloadedType.__init__(self, *args, **kwargs)

    def create_block_variable(self):
        block_variable = OverloadedType.create_block_variable(self)
        block_variable.floating_type = True
        return block_variable

    def _ad_will_add_as_dependency(self):
        if self._ad_floating_active:
            with FloatingType.stop_floating(self):
                self._ad_annotate_block()
        return True

    def _ad_will_add_as_output(self):
        if self._ad_floating_active:
            with FloatingType.stop_floating(self):
                self._ad_annotate_output_block()
        return True

    def _ad_annotate_block(self):
        if self.block_class is None:
            return

        if not self.annotate_tape:
            return

        tape = get_working_tape()
        block = self.block_class(*self._ad_args, **self._ad_kwargs)
        self.block = block
        tape.add_block(block)
        block.add_output(self.block_variable)

        # Need to create a new block output for future use.
        self.create_block_variable()

    def _ad_annotate_output_block(self):
        if self.output_block_class is None:
            return

        if not self.annotate_tape:
            return

        tape = get_working_tape()
        block = self.output_block_class(self, *self._ad_output_args, **self._ad_output_kwargs)
        self.output_block = block
        tape.add_block(block)
        for output in self._ad_outputs:
            block.add_output(output.create_block_variable())

    class stop_floating(object):
        def __init__(self, obj):
            self.obj = obj

        def __enter__(self):
            self.obj._ad_floating_active = False

        def __exit__(self, *args):
            self.obj._ad_floating_active = True
