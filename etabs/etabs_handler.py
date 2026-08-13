import os
import sys
import comtypes.client
import json
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr
from typing import Optional, Any, Dict

class ETABSHandler(BaseModel):
    # Pydantic v2 config (optional helpers)
    model_config = ConfigDict(
        validate_assignment=True,   # re-validate when you set attrs later
        extra='forbid',             # disallow undeclared fields
         # populate_by_name=True,    # enable using field *names* when aliases exist (see note below)
    )

    attach: bool = False
    specify_path: bool = False
    program_path: Optional[str] = None  # no alias
    visible: bool = True

    _helper: Any = PrivateAttr(default=None)
    _etabs_object: Any = PrivateAttr(default=None)
    _sap_model: Any = PrivateAttr(default=None)
    _started_here: bool = PrivateAttr(default=False)

    def connect_to_etabs(self):
        # Honor the constructor parameters (previously hardcoded — bug).
        AttachToInstance = self.attach
        SpecifyPath = self.specify_path or bool(self.program_path)
        ProgramPath = self.program_path or R'C:\Program Files\Computers and Structures\ETABS 23\ETABS.exe'

        # full path to the model
        # set it to the desired path of your model
        APIPath = R'C:\CSi_ETABS_API_Example'
        if not os.path.exists(APIPath):
            try:
                os.makedirs(APIPath)
            except OSError:
                pass
        ModelPath = APIPath + os.sep + 'API_1-001.edb'
        comtypes.CoInitialize()

        # create API helper object
        self._helper = comtypes.client.CreateObject('ETABSv1.Helper')
        self._helper = self._helper.QueryInterface(comtypes.gen.ETABSv1.cHelper)

        if AttachToInstance:
            # attach to a running instance of ETABS
            try:
                # get the active ETABS object
                self._etabs_object = self._helper.GetObject("CSI.ETABS.API.ETABSObject")
            except (OSError, comtypes.COMError):
                print("No running instance of the program found or failed to attach.")
                sys.exit(-1)
        else:
            if SpecifyPath:
                try:
                    # 'create an instance of the ETABS object from the specified path
                    self._etabs_object = self._helper.CreateObject(ProgramPath)
                except (OSError, comtypes.COMError):
                    print("Cannot start a new instance of the program from " + ProgramPath)
                    sys.exit(-1)
            else:
                try:
                    # create an instance of the ETABS object from the latest installed ETABS
                    self._etabs_object = self._helper.CreateObjectProgID("CSI.ETABS.API.ETABSObject")
                except (OSError, comtypes.COMError):
                    print("Cannot start a new instance of the program.")
                    sys.exit(-1)

            # start ETABS application
            self._etabs_object.ApplicationStart()

        # create SapModel object
        self._sap_model = self._etabs_object.SapModel

        # initialize model
        self._sap_model.InitializeNewModel(Units=comtypes.gen.ETABSv1.eUnits_kN_m_C)

        # create new blank model
        ret = self._sap_model.File.NewBlank()


    def print_model_name(self):
        model_name = self._sap_model.GetModelFilename()
        print(model_name)

    def disconnect_from_etabs(self, close=False):
        if close:
            self._etabs_object.ApplicationExit(False)
        _sap_model = None
        _etabs_object = None

    def init_default_model(self):
        # define material property
        MATERIAL_CONCRETE = 2
        ret = self._sap_model.PropMaterial.SetMaterial('CONC', MATERIAL_CONCRETE)

        # assign isotropic mechanical properties to material
        ret = self._sap_model.PropMaterial.SetMPIsotropic('CONC', 3600, 0.2, 0.0000055)

        # define rectangular frame section property
        ret = self._sap_model.PropFrame.SetRectangle('R1', 'CONC', 12, 12)

        # define frame section property modifiers
        ModValue = [1000, 0, 0, 1, 1, 1, 1, 1]
        ret = self._sap_model.PropFrame.SetModifiers('R1', ModValue)

        # switch to k-ft units
        Kn_m_C = 6
        ret = self._sap_model.SetPresentUnits(Kn_m_C)

    def define_node(self, x , y, z : float , point_name: str):
        #x, y, z = float(n["x"]), float(n["y"]), float(n["z"])
        #node_id_to_xyz[n["id"]] = (x, y, z)

        try:
            ret, point_name = self._sap_model.PointObj.AddCartesian(x, y, z, "")
        except Exception:
            point_name = ""
            ret = self._sap_model.PointObj.AddCartesian(x, y, z, point_name)
        # if you have a _check wrapper: _check(ret, "PointObj.AddCartesian")
       #node_id_to_point[n["id"]] = point_name

    def define_nodes(self, structure_json : Dict):
        for n in structure_json.get("nodes", []):
            x, y, z = float(n["x"]), float(n["y"]), float(n["z"])
            point_name = n["id"]
            self.define_node(x, y, z, point_name)

    def define_frame(self, x_start , y_start , z_start , x_end , y_end , z_end : float, frame_name : str):
        self._sap_model.FrameObj.AddByCoord(x_start, y_start, z_start, x_end, y_end, z_end, frame_name, 'R1', '1', 'Global')

    def define_frames(self, structure_json: Dict[str, Any]) -> None:
        # Build an index: node_id -> node_dict
        nodes_list = structure_json.get("nodes", [])
        node_by_id = {n["id"]: n for n in nodes_list if "id" in n}

        for m in structure_json.get("members", []):
            start_id = m["start_node"]
            end_id = m["end_node"]
            frame_name = m["id"]
            try:
                n1 = node_by_id[start_id]
                n2 = node_by_id[end_id]

                x_start, y_start, z_start = float(n1["x"]), float(n1["y"]), float(n1["z"])
                x_end, y_end, z_end = float(n2["x"]), float(n2["y"]), float(n2["z"])
                self.define_frame(x_start, y_start, z_start, x_end, y_end, z_end, frame_name)
            except KeyError as e:
                raise KeyError(f"Missing node or coordinate {e!s} while processing member {m}") from e
            except (TypeError, ValueError) as e:
                raise ValueError(f"Non-numeric coordinate for nodes {start_id}/{end_id}") from e

            # Use the coordinates as needed, e.g.:
            # self.add_frame((x_start, y_start, z_start), (x_end, y_end, z_end), member=m)