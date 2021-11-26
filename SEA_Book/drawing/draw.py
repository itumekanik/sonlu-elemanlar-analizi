from .plot import Plot

class TriangleMaps:
    def __init__(self, *node_indices):
        self.map = [ijk
                    for ijk in node_indices]

class LineMaps:
    def __init__(self, *node_indices):
        self.map = [ijk
                    for ijk in node_indices]


class Drawer:
    def getDescriptor(self, dtype):
        _desc = [v
                 for k, v in self.__class__.__dict__.items()
                 if isinstance(v, dtype)]

        if len(_desc) == 0:
            print("Error: Missing Descriptor", dtype)
        return _desc[-1]

    def __init__(self,
                 elements,
                 nodes,
                 on1,
                 on2,
                 connectivity_name="conn",
                 mesh=False,
                 name="",
                 unit="",
                 lookat= [0, 0, 1]):

        self.element_class = elements.values()
        self.node_class = nodes.values()
        self.on1 = on1
        self.on2 = on2
        self._conn_name = connectivity_name
        self.mesh = mesh
        self.elements = (e for e in self.element_class)
        self.name = name
        self.unit = unit
        self.lookat = lookat
        for i, n in enumerate(self.node_class): n._index = i
        for i, e in enumerate(self.element_class): e._index = i


    def __getitem__(self, item):
        self.elements = (e for e in self.element_class if item(e))
        return self

    def __getattr__(self, item):
        self.function_name = item
        return self

    def __call__(self, *args, **kwargs):
        return self._plot()


    def _plot(self):
        _XYZ = self.on1
        _xyz = self.on2
        _mesh = self.mesh
        _trigs = self.getDescriptor(TriangleMaps).map
        _lines = self.getDescriptor(LineMaps).map

        #from .old_descriptors import fwConnectivity

        #from ..conn.connectivity import Connectivity

        #_conn_name = [v for k, v in self.element_class.__dict__.items() if isinstance(v, Connectivity)][0].fwname

        # _conn_name = "nodes"

        return Plot(self.elements,
                    self.function_name,
                    _trigs, _lines,
                    self.node_class,
                    _XYZ,
                    _xyz,
                    conn_name=self._conn_name,
                    mesh=_mesh,
                    name=self.name,
                    unit=self.unit,
                    lookat=self.lookat)


