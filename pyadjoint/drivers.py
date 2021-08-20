from .enlisting import Enlist
from .tape import get_working_tape, stop_annotating


def compute_jacobian_adjoint_action(J, m, v, options=None, tape=None):
    """
    Compute the adjoint Jacobian of function J with respect to the initialisation value of m,
    that is the value of m at its creation, in a direction v.

    Args:
        J (list of instance of OverloadedType):  The (list of) functions.
        m (list or instance of Control): The (list of) controls.
        v (list or instance of the function type): The direction in which to compute the adjoint Jacobian.
        options (dict): A dictionary of options. To find a list of available options
            have a look at the specific control type.
        tape: The tape to use. Default is the current tape.

    Returns:
        OverloadedType: The derivative with respect to the control. Should be an instance of the same type as
            the control.
    """
    options = {} if options is None else options
    tape = get_working_tape() if tape is None else tape
    tape.reset_variables()
    J = Enlist(J)
    m = Enlist(m)
    v = Enlist(v)
    for (f,adj_value) in zip(J,v):
        f.adj_value = adj_value

    with stop_annotating():
        with tape.marked_nodes(m):
            tape.evaluate_adj(markings=True)

    grads = [i.get_derivative(options=options) for i in m]
    return m.delist(grads)


def compute_jacobian_action(J, m, m_dot, options=None, tape=None):
    """
    Compute the Jacobian of function J with respect to the initialisation value of m,
    that is the value of m at its creation, in a direction m_dot
    Args:
        J (list of instance of OverloadedType):  The (list of) functions.
        m (list or instance of Control): The (list of) controls.
        m_dot (list or instance of the control type): The direction in which to compute the Jacobian.
        options (dict): A dictionary of options. To find a list of available options
            have a look at the specific control type.
        tape: The tape to use. Default is the current tape.
    Returns:
        OverloadedType: The derivative with respect to the control. Should be an instance of the same type as
            the control.
    """
    options = {} if options is None else options
    tape = get_working_tape() if tape is None else tape
    tape.reset_variables()
    J = Enlist(J)
    m = Enlist(m)
    m_dot = Enlist(m_dot)
    for i, value in enumerate(m_dot):
        m[i].tlm_value = m_dot[i]
    with stop_annotating():
        tape.evaluate_tlm()
    jacs = [i.block_variable.tlm_value(options=options) for i in J]
    return J.delist(jacs)


def compute_gradient(J, m, options=None, tape=None, adj_value=1.0):
    return compute_jacobian_adjoint_action(J, m, adj_value, options, tape)




def compute_hessian(J, m, m_dot, options=None, tape=None, adj_value=1.0):
    """
    Compute the Hessian of J in a direction m_dot at the current value of m
    Args:
        J (AdjFloat):  The objective functional.
        m (list or instance of Control): The (list of) controls.
        m_dot (list or instance of the control type): The direction in which to compute the Hessian.
        options (dict): A dictionary of options. To find a list of available options
            have a look at the specific control type.
        tape: The tape to use. Default is the current tape.
    Returns:
        OverloadedType: The second derivative with respect to the control in direction m_dot. Should be an instance of
            the same type as the control.
    """
    
    if tape is None:
        tape = get_working_tape() 
    else:
        tape

    options = {} if options is None else options
    tape.reset_tlm_values()
    tape.reset_hessian_values()
    m = Enlist(m)
    m_dot = Enlist(m_dot)
    for i, value in enumerate(m_dot):
        m[i].tlm_value = m_dot[i]
    with stop_annotating():
        tape.evaluate_tlm()
    J.block_variable.hessian_value = 0.0
    with stop_annotating():
        with tape.marked_nodes(m):
            tape.evaluate_hessian(markings=True)
    r = [v.get_hessian(options=options) for v in m]
    return m.delist(r)