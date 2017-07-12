from .tape import get_working_tape, annotate_tape
from .block_output import BlockOutput


class OverloadedType(object):
    """Base class for OverloadedType types.

    The purpose of each OverloadedType is to extend a type such that
    it can be referenced by blocks as well as overload basic mathematical
    operations such as __mul__, __add__, where they are needed.

    Abstract methods:
        :func:`adj_update_value`

    """
    def __init__(self, *args, **kwargs):
        # TODO: Do we actually need to store the tape? I don't think this is used at all.
        tape = kwargs.pop("tape", None)

        if tape:
            self.tape = tape
        else:
            self.tape = get_working_tape()

        self.original_block_output = self.create_block_output()

    def create_block_output(self):
        block_output = BlockOutput(self)
        self.set_block_output(block_output)
        return block_output

    def set_block_output(self, block_output):
        self.block_output = block_output

    def get_block_output(self):
        return self.block_output

    def get_adj_output(self):
        return self.original_block_output.get_adj_output()

    def set_initial_adj_input(self, value):
        self.block_output.set_initial_adj_input(value)

    def set_initial_tlm_input(self, value):
        self.original_block_output.set_initial_tlm_input(value)

    def reset_variables(self):
        self.original_block_output.reset_variables()

    def get_derivative(self, options={}):
        # TODO: Decide on naming here.
        # Basically the method should implement a way to convert
        # the adj_output to the same type as `self`.
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


class FloatingType(OverloadedType):
    def __init__(self, *args, **kwargs):
        self.block_class = kwargs.pop("block_class", None)
        self._ad_args = kwargs.pop("_ad_args", [])
        self._ad_kwargs = kwargs.pop("_ad_kwargs", {})
        self._ad_floating_active = kwargs.pop("_ad_floating_active", False)
        self.annotate_tape = annotate_tape(kwargs)
        self.block = None
        OverloadedType.__init__(self, *args, **kwargs)

    def create_block_output(self):
        block_output = OverloadedType.create_block_output(self)
        block_output.floating_type = True
        return block_output

    def _ad_annotate_block(self):
        if self.block_class is None:
            return

        if not self.annotate_tape:
            return

        tape = get_working_tape()
        block = self.block_class(self, *self._ad_args, **self._ad_kwargs)
        self.block = block
        tape.add_block(block)
        block.add_output(self.block_output)

        # Need to create a new block output for future use.
        self.create_block_output()


