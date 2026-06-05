import numpy as np
from node import Node
from reservoirs import Reservoir


class Network():
    """
    This class defines a Network. A network is a collection of Nodes and it is
    used to route the fluxes from upstream to downstream. A network must be a
    tree.
    """

    def __init__(self, nodes, topology):
        """
        This is the initializer of the class Network.

        Parameters
        ----------
        nodes : list(superflexpy.framework.node.Node)
            List of nodes that belongs to the network. The order is not
            important.
        topology : dict(str : str)
            Topology of the network. Keys are the id of the nodes and values
            are the id of the downstream node the key. Since the network must
            be a tree, each key has only one downstream element
        """

        self._error_message = 'module : superflexPy, Network ,'
        self._error_message += ' Error message : '

        for n in nodes:
            if not isinstance(n, Node):
                message = '{}units must be instance of the Unit class'.format(self._error_message)
                raise TypeError(message)

        self._nodes = nodes
        self._downstream = topology

        self._build_network()

    def run_network(self):
        """
        This method solves the network, solving each node and putting together
        their outputs according to the topology of the network.

        Returns
        -------
        :dict(str : list(numpy.ndarray))
            Dictionary containig the outputs of all the nodes.
        
            The outputs are: Tuple(numpy.array, numpy.array, numpy.ndarray, numpy.ndarray)
                Tuple with:
                - the output flux timeseries (index 0) as numpy.array of length t;
                - the output concentration timeseries (index 1) as numpy.array of length t;
                - the output water TTDs (index 2, timeseries of output TTD) as numpy.ndarray of dimension t x T;
                - the output mass TTDs (index 3, timeseries of output TTD) as numpy.ndarray of dimension t x T.
        """

        # Keep track of the solved nodes
        solved = {k: False for k in self._upstream.keys()}
        output = {}

        # Solve first the headwater nodes
        for n in self._headwater:
            output[n] = self._nodes[self._nodes_pointer[n]].run_node()
            solved[n] = True

        if len(self._nodes) != len(self._headwater):
            completed = False
        else:
            completed = True

        # Solve for progressively more downstream nodes
        while not completed:
            for n in self._upstream.keys():
                if not solved[n]:
                    # Check if all the upstream nodes have been solved
                    solvable = True
                    for n_up in self._upstream[n]:
                        if not solved[n_up]:
                            solvable = False
                    if solvable:
                        # Solve the current node
                        q, c, ttd_q, ttd_m = self._nodes[self._nodes_pointer[n]].run_node()
                        w = self._nodes[self._nodes_pointer[n]].area / self._total_area[n]

                        # Combine with fluxes from upstream nodes
                        out_q = q * w
                        out_c = c * q * w if c is not None else None
                        out_ttd_q = ttd_q * q[:, np.newaxis] * w if ttd_q is not None else None
                        out_mflux = c * q * w if c is not None else None
                        out_ttd_m = ttd_m * c[:, np.newaxis] * q[:, np.newaxis] * w if ttd_m is not None else None
                        
                        for n_up in self._upstream[n]:
                            q_up, c_up, ttd_q_up, ttd_m_up = (
                                self._nodes[self._nodes_pointer[n_up]].external_routing(output[n_up])
                                )
                            w_up = self._total_area[n_up] / self._total_area[n]

                            out_q += q_up * w_up
                            out_c = out_c + c_up * q_up * w_up if (out_c is not None and c_up is not None) else None
                            out_ttd_q = out_ttd_q + ttd_q_up * q_up[:, np.newaxis] * w_up if (out_ttd_q is not None and ttd_q_up is not None) else None
                            out_mflux = out_mflux + c_up * q_up * w_up if (out_mflux is not None and c_up is not None) else None
                            out_ttd_m = out_ttd_m + ttd_m_up * c_up[:, np.newaxis] * q_up[:, np.newaxis] * w_up if (out_ttd_m is not None and ttd_m_up is not None) else None

                        if out_c is not None: out_c /= out_q
                        if out_ttd_q is not None: out_ttd_q /= out_q[:, np.newaxis]
                        if out_ttd_m is not None: out_ttd_m /= out_mflux[:, np.newaxis]

                        output[n] = tuple((out_q, out_c, out_ttd_q, out_ttd_m))
                        solved[n] = True

                        if self._downstream[n] is None:
                            completed = True

        return output

    def _build_network(self):
        """
        Constructs the internal data structures needed to solve the network.

        This method is a preprocessing step that must be called before running
        the network. Starting from the user-defined downstream connectivity
        (_downstream), it derives the full graph topology and computes the total
        upstream contributing area for each node via a topological traversal.

        Constructs
        ----------
        _upstream : dict(Node, list(Node) or None)
            Inverse of _downstream. Maps each node to the list of nodes that
            drain directly into it, or None if the node is a headwater.

        _headwater : list(Node)
            Nodes with no upstream contributors. These receive only direct
            rainfall inputs and never lateral inflow from other nodes.

        _nodes_pointer : dict(str, int)
            Maps each node's string identifier to its index in _nodes, allowing
            O(1) node lookup by id.

        _total_area : dict(Node, float)
            Maps each node to its total upstream contributing area, i.e. the
            sum of its own area and the areas of all nodes draining into it,
            directly or indirectly. Computed via a topological traversal
            starting from the headwaters and proceeding downstream, so that
            each node is only processed once all its upstream nodes are solved.
            The outlet node (with _downstream = None) holds the total
            nchment area.
        """

        # Find the upstream nchments (dict: each Node is a key and upstream Nodes list as value)
        self._upstream = {k: [] for k in self._downstream.keys()}
        for n in self._downstream.keys():
            if self._downstream[n] is not None:
                self._upstream[self._downstream[n]].append(n)

        for n in self._upstream.keys():
            if len(self._upstream[n]) == 0:
                self._upstream[n] = None

        # Find the headwater
        self._headwater = [k for k in self._upstream.keys() if self._upstream[k] is None]

        # Build the map from id to index
        self._nodes_pointer = {n.id: i for i, n in enumerate(self._nodes)}

        # Calculate the total area
        self._total_area = {}
        solved = {k: False for k in self._upstream.keys()}

        # First the headwaters
        for n in self._headwater:
            self._total_area[n] = self._nodes[self._nodes_pointer[n]].area
            solved[n] = True

        if len(self._nodes) != len(self._headwater):
            completed = False
        else:
            completed = True

        while not completed:
            for n in self._upstream.keys():
                if not solved[n]:
                    # Check if all the upstreams have been solved
                    solvable = True
                    for n_up in self._upstream[n]:
                        if not solved[n_up]:
                            solvable = False
                    if solvable:
                        # Solve the current nchment
                        area = self._nodes[self._nodes_pointer[n]].area

                        for n_up in self._upstream[n]:
                            area += self._total_area[n_up]

                        self._total_area[n] = area
                        solved[n] = True

                        if self._downstream[n] is None:
                            completed = True
    
    def _check_flag_consistency(self):
        """
        Verifies that all Reservoir elements in the network have consistent
        computation flags. This method must be called after _build_network()
        and before run_network(), to ensure that the None-propagation logic
        in run_network() behaves consistently across all nodes.

        The following flags are checked for consistency across all reservoirs:
            - concentration_computation
            - age_computation
            - mass_tracking_computation

        The following implied constraints are also verified:
            - mass_tracking_computation=True requires age_computation=True
            - mass_tracking_computation=True requires concentration_computation=True

        Raises
        ------
        ValueError
            If any flag is inconsistent across reservoirs, or if an implied
            constraint is violated. The error message identifies which reservoir
            violates the constraint and what the expected value is.
        """

        # Collect all Reservoir elements across the network,
        # traversing Network -> Nodes -> Units -> Layers -> Elements
        reservoirs = []
        for node in self._nodes:
            for unit in node.units:
                for layer in unit._layers:
                    for element in layer:
                        if isinstance(element, Reservoir):
                            reservoirs.append(element)

        if len(reservoirs) == 0:
            return

        # Use the first reservoir as the reference
        ref = reservoirs[0]
        ref_flags = {
            'concentration_computation': ref.concentration_computation,
            'age_computation':           ref.age_computation,
            'mass_tracking_computation': ref.mass_tracking_computation,
        }

        # Check consistency of all reservoirs against the reference
        for res in reservoirs[1:]:
            for flag, ref_val in ref_flags.items():
                if getattr(res, flag) != ref_val:
                    raise ValueError(
                        f'Inconsistent flag "{flag}" in Reservoir "{res._id}": '
                        f'expected {ref_val} (as in Reservoir "{ref._id}"), '
                        f'got {getattr(res, flag)}.'
                    )

        # Check implied constraints (sufficient to check on reference
        # since all reservoirs are consistent at this point)
        if ref_flags['mass_tracking_computation']:
            if not ref_flags['age_computation']:
                raise ValueError(
                    'mass_tracking_computation=True requires age_computation=True '
                    'across all reservoirs.'
                )
            if not ref_flags['concentration_computation']:
                raise ValueError(
                    'mass_tracking_computation=True requires concentration_computation=True '
                    'across all reservoirs.'
                )
