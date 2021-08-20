from .drivers import compute_jacobian_action, compute_jacobian_adjoint_action, compute_gradient, compute_hessian
from .enlisting import Enlist
from .tape import get_working_tape, stop_annotating, no_annotations

class ReducedFunction(object):
    """Class representing the reduced function.
    A reduced function maps a control value to the provided values.
    It may also be used to compute the Jacobian of the function with
    respect to the control.
    Args:
        function (list[OverloadedType]): A list of  OverloadedType instances.
            It is also possible to supply a single OverloadedTpe instance
            instead of a list.  This should be the return value of the function
            you want to reduce.
        controls (list[Control]): A list of Control instances, which you want
            to map to the function. It is also possible to supply a single Control
            instance instead of a list.
    """
    def __init__(self, functions, controls, tape=None,
                 eval_cb_pre=lambda *args: None,
                 eval_cb_post=lambda *args: None,
                 jacobian_cb_pre=lambda *args: None,
                 jacobian_cb_post=lambda *args: None,
                 jacobian_adj_cb_pre=lambda *args: None,
                 jacobian_adj_cb_post=lambda *args: None):
        self.functions = Enlist(functions)
        self.tape = get_working_tape() if tape is None else tape
        self.controls = Enlist(controls)
        self.eval_cb_pre = eval_cb_pre
        self.eval_cb_post = eval_cb_post
        self.jacobian_cb_pre = jacobian_cb_pre
        self.jacobian_cb_post = jacobian_cb_post
        self.jacobian_adj_cb_pre = jacobian_adj_cb_pre
        self.jacobian_adj_cb_post = jacobian_adj_cb_post
    @no_annotations
    def jacobian_action(self, m_dot, options={}):
        """Returns the action of the jacobian of the function w.r.t. the control on a vector m_dot.
        Using the forward method, the Jacobian of the function with
        respect to the control, around the last supplied value of the control,
        is computed and returned.
        Args:
            m_dot ([OverloadedType]): The direction in which to compute the
                action of the Jacobian.
            options (dict): A dictionary of options. To find a list of available options
                have a look at the specific control type.
        Returns:
            OverloadedType: The action of the Jacobian with respect to the controls on a vector m_dot.
                Should be an instance of the same type as the functions.
        """
        # Call callback
        values = [c.data() for c in self.controls]
        func_values = [f.block_variable.checkpoint for f in self.functions]
        self.jacobian_cb_pre(self.controls.delist(values))
        jacobian = compute_jacobian_action(self.functions,
                                           self.controls,
                                           m_dot,
                                           options=options,
                                           tape=self.tape)
        # Call callback
        self.jacobian_cb_post(self.functions.delist(func_values),
                              self.controls.delist(jacobian),
                              self.controls.delist(values))
        return self.functions.delist(jacobian)



    @no_annotations
    def jacobian_adjoint_action(self, v, options={}):
        """Returns the adjoint action of the jacobian of the functional w.r.t. the control on a vector v.
        Using the adjoint method, the adjoint Jacobian of the functional with
        respect to the control, around the last supplied value of the control,
        is computed and returned.
        Args:
            v ([OverloadedType]): The direction in which to compute the
                adjoint action of the Jacobian.
            options (dict): A dictionary of options. To find a list of available options
                have a look at the specific control type.
        Returns:
            OverloadedType: The action of the Jacobian with respect to the controls on a vector v.
                Should be an instance of the same type as the controls.
        """
        # Call callback
        values = [c.data() for c in self.controls]
        func_values = [f.block_variable.checkpoint for f in self.functions]
        self.jacobian_adj_cb_pre(self.controls.delist(values))
        adj_jacobian = compute_jacobian_adjoint_action(self.functions,
                                                       self.controls,
                                                       v,
                                                       options=options,
                                                       tape=self.tape)
        # Call callback
        self.jacobian_adj_cb_post(func_values,
                                  self.controls.delist(adj_jacobian),
                                  self.controls.delist(values))
        return self.controls.delist(adj_jacobian)

    @no_annotations
    def __call__(self, values):
        """Computes the reduced functional with supplied control value.
        Args:
            values ([OverloadedType]): If you have multiple controls this should be a list of
                new values for each control in the order you listed the controls to the constructor.
                If you have a single control it can either be a list or a single object.
                Each new value should have the same type as the corresponding control.
        Returns:
            :obj:`OverloadedType`: The computed value.
        """
        values = Enlist(values)
        if len(values) != len(self.controls):
            raise ValueError("values should be a list of same length as controls.")
        # Call callback.
        self.eval_cb_pre(self.controls.delist(values))
        for i, value in enumerate(values):
            self.controls[i].update(value)
        self.tape.reset_blocks()
        blocks = self.tape.get_blocks()
        with self.marked_controls():
            with stop_annotating():
                for i in range(len(blocks)):
                    blocks[i].recompute()
        func_values = [f.block_variable.checkpoint for f in self.functions]
        # Call callback
        self.eval_cb_post(self.functions.delist(func_values), self.controls.delist(values))
        return self.functions.delist(func_values)
    def optimize_tape(self):
        self.tape.optimize(
            controls=self.controls,
            functionals=self.functions
        )
    def marked_controls(self):
        return marked_controls(self)


class ReducedFunctional(ReducedFunction):
    """Class representing the reduced functional.

    A reduced functional maps a control value to the provided functional.
    It may also be used to compute the derivative of the functional with
    respect to the control.

    Args:
        functional (:obj:`OverloadedType`): An instance of an OverloadedType,
            usually :class:`AdjFloat`. This should be the return value of the
            functional you want to reduce.
        controls (list[Control]): A list of Control instances, which you want
            to map to the functional. It is also possible to supply a single Control
            instance instead of a list.

    """

    def __init__(self, functional, controls, scale=1.0, tape=None,
                 eval_cb_pre=lambda *args: None,
                 eval_cb_post=lambda *args: None,
                 derivative_cb_pre=lambda *args: None,
                 derivative_cb_post=lambda *args: None,
                 hessian_cb_pre=lambda *args: None,
                 hessian_cb_post=lambda *args: None):
        self.scale = scale
        super(ReducedFunctional,self).__init__(functional, controls,
                                               tape, eval_cb_pre=eval_cb_pre,
                                               jacobian_adj_cb_pre=derivative_cb_pre,
                                               jacobian_adj_cb_post=derivative_cb_post)
        self.fnl_eval_cb_post = eval_cb_post
        self.derivative_cb_pre = derivative_cb_pre
        self.derivative_cb_post = derivative_cb_post
        self.hessian_cb_pre = hessian_cb_pre
        self.hessian_cb_post = hessian_cb_post

    @property
    def functional(self):
        return self.functions[0]

    def derivative(self, options={}):
        """Returns the derivative of the functional w.r.t. the control.

        Using the adjoint method, the derivative of the functional with
        respect to the control, around the last supplied value of the control,
        is computed and returned.

        Args:
            options (dict): A dictionary of options. To find a list of available options
                have a look at the specific control type.

        Returns:
            OverloadedType: The derivative with respect to the control.
                Should be an instance of the same type as the control.

        """
        return self.jacobian_adjoint_action(self.scale, options)
        


    @no_annotations
    def hessian(self, m_dot, options={}):
        """Returns the action of the Hessian of the functional w.r.t. the control on a vector m_dot.

        Using the second-order adjoint method, the action of the Hessian of the
        functional with respect to the control, around the last supplied value
        of the control, is computed and returned.

        Args:
            m_dot ([OverloadedType]): The direction in which to compute the
                action of the Hessian.
            options (dict): A dictionary of options. To find a list of
                available options have a look at the specific control type.

        Returns:
            OverloadedType: The action of the Hessian in the direction m_dot.
                Should be an instance of the same type as the control.
        """
        # Call callback
        values = [c.data() for c in self.controls]
        self.hessian_cb_pre(self.controls.delist(values))

        r = compute_hessian(self.functional, self.controls, m_dot, options=options, tape=self.tape)

        # Call callback
        self.hessian_cb_post(self.functional.block_variable.checkpoint,
                             self.controls.delist(r),
                             self.controls.delist(values))

        return self.controls.delist(r)

    @no_annotations
    def __call__(self, values):
        """Computes the reduced functional with supplied control value.

        Args:
            values ([OverloadedType]): If you have multiple controls this should be a list of
                new values for each control in the order you listed the controls to the constructor.
                If you have a single control it can either be a list or a single object.
                Each new value should have the same type as the corresponding control.

        Returns:
            :obj:`OverloadedType`: The computed value. Typically of instance
                of :class:`AdjFloat`.

        """
        values = Enlist(values)
        unscaled = super(ReducedFunctional,self).__call__(values)
        func_value = self.scale * unscaled

        # Call callback
        self.fnl_eval_cb_post(func_value, self.controls.delist(values))

        return func_value

    

class marked_controls(object):
    def __init__(self, rf):
        self.rf = rf

    def __enter__(self):
        for control in self.rf.controls:
            control.mark_as_control()

    def __exit__(self, *args):
        for control in self.rf.controls:
            control.unmark_as_control()
