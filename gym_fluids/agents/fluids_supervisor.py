import fluids

# FLUIDS supervisor agent.

def fluids_supervisor(obs, info_dict={}):
    if "supervisor_action" not in info_dict:
        print("ERROR: You should pass the info_dict from the FLUIDS gym environment here to plan")
        exit(1)
    return info_dict["supervisor_action"]
