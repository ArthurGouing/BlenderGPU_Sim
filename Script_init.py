#### Blender Import #### 
import bpy
from bpy.app.handlers import persistent
from bpy.utils import register_class, unregister_class
from mathutils import Vector

### Library Import ####
import sys, os
from time import perf_counter
# Set global variables

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)
print(dir, bpy.data.filepath)
print(sys.path)

### Local Import ###
import simulator
import parameter

# this next part forces a reload in case you edit the source after you first start the blender session
import imp
imp.reload(simulator)
imp.reload(parameter)

from simulator import Simulator
from parameter import cloth_sim
# sim = Simulator()

# sim.n = 200

# @persistent
def step_forward(scene):
    """
    Rules of computation according to frame change:
     - if we go to the next frame:     Compute one Simulation loop
     - if we go to the previous frame: Do nothing
     - if we jump to a father frame:   Reset the computation
     - if we compile and go to the next step: the computation is'nt init --> Faire l'init dans la compile ?
    """
    # Init local variables
    obj = scene.cloth_simulator.obj
    delta_frame = scene.frame_current - scene.frame_previous
    scene.frame_previous = scene.frame_current
    print("delta frame:", delta_frame)

    # Compute
    if cloth_sim.isnot_init:
        t_init = perf_counter()
        cloth_sim.initialize_from_obj(scene.cloth_simulator.obj)
        print(f"Init time: {(perf_counter()-t_init)*1e3:.3f} ms")
    if delta_frame == 1:
        print("step formward")
        print(f"n substep = ", cloth_sim.substeps)
        t_forward = perf_counter()
        cloth_sim.frame_forward()
        print(f"Forward time: {(perf_counter()-t_forward)*1e3:.3f} ms")

        t_update = perf_counter()
        cloth_sim.update_vertices(obj)
        print(f"Update time: {(perf_counter()-t_update)*1e3:.3f} ms")
    elif delta_frame != -1 and delta_frame != 0:
        # Do the pre-process part and reset  GPU buffers
        print("Reset the simulation")
        t_reset = perf_counter()
        cloth_sim.reset()
        print(f"Reset time: {(perf_counter()-t_reset)*1e3:.3f} ms")
        t_update = perf_counter()
        cloth_sim.update_vertices(obj)
        print(f"Update time: {(perf_counter()-t_update)*1e3:.3f} ms")


class ClothPanel(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "Cloth Simulator"
    bl_idname = "VIEW3D_PT_ClothSimu"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GPU Simulation"
    bl_context = "objectmode"

    def draw(self, context):
        layout = self.layout
        obj = context.object
        sim = context.scene.cloth_simulator

        layout.label(text="Powered by Taichi", icon='MOD_CLOTH')
        layout.label(text="Simulation parameters")
        layout.prop(sim, "obj")
        layout.prop(sim, "dt")
        col = layout.column()
        col.prop(sim,'gravity')
        layout.prop(sim, "k")
        layout.prop(sim, "alpha")
        layout.prop(sim, "drag")

        # layout.operator("object.add_simulator", text="Init Simulator", icon="ADD")
        # layout.prop(context, "object", text="Cloth Object")
        return


class ClothSimulationProperty(bpy.types.PropertyGroup):
    """
    Custom property structure.
    Belong to : bpy.types.Scene.cloth_simulator
    TODO: a function for each property to modify the Simulator attribute directly when the param is changed
    """
    obj:     bpy.props.PointerProperty(type=bpy.types.Object, name="Cloth mesh")
    dt:      bpy.props.FloatProperty(name="Delta Time", default=cloth_sim.dt, precision=6)
    fps:     bpy.props.IntProperty(name="FPS", default=24)
    gravity: bpy.props.FloatVectorProperty(name="Gravity", subtype='ACCELERATION', default=Vector(cloth_sim.gravity))
    k:       bpy.props.FloatProperty(name="Spring rigidity")
    alpha:   bpy.props.FloatProperty(name="Spring damping")
    drag:    bpy.props.FloatProperty(name="Air drag")


classes = [ClothPanel, ClothSimulationProperty]

def register():
    print("register")
    ### Register Class ###
    for cls in classes:
        print("register: ", cls.__name__)
        register_class(cls)
    ### Add handlers
    bpy.app.handlers.frame_change_pre.append(step_forward)
    ### Add Cloth properties to scene
    bpy.types.Scene.cloth_simulator = bpy.props.PointerProperty(type=ClothSimulationProperty)
    bpy.types.Scene.frame_previous = bpy.props.IntProperty(name="p_frame", default=0)

def unregister():
    print("unregister")
    ### Delete handlers
    bpy.app.handlers.frame_change_pre.remove(
            bpy.app.handlers.frame_change_pre[0]
    )
    ### Unregister Classes ###
    for cls in reversed(classes):
        print(cls.__name__, cls)
        unregister_class(cls)
    ### Del Cloth properties to scene
    del bpy.types.scene.cloth_simulator
    del bpy.types.Scene.frame_previous


    
if __name__ == "__main__":
    # simulator = Simulator()
    try:
        unregister()
    except:
        print("unregister failed")
        pass
    register()
    print("Done loading script")