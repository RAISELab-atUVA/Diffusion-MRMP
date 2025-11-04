from pyomo.environ import *
import numpy as np
import torch
import copy
import pickle


from torch_robotics import environments, robots
from smd.utils.loading import load_params_from_yaml




def solve_L_aug(x, agent_starts, agent_goals, agent_rads, max_speeds, obs_pos, obs_rads, horizons, nu_o, nu_a, rho, init_traj4proj_array, init_d_o, init_d_a):
    num_agents = agent_starts.shape[0]
    num_obs = obs_pos.shape[0]
    max_speeds = max_speeds * np.ones(num_agents)



    model = ConcreteModel()

    # Define sets
    model.I = RangeSet(0, horizons - 1)                # Time steps
    model.J = RangeSet(0, num_agents - 1)              # Agents
    model.K = RangeSet(0, num_obs - 1)                 # Obstacles
    model.Dim = RangeSet(0, 1)                         # Dimensions (x and y)
    model.AgentPairs = Set(initialize=[(j, k) for j in range(num_agents) for k in range(num_agents) if k > j])

    initial_p_out = {(i, j, k): init_traj4proj_array[i, j, k] for i in range(horizons) for j in range(num_agents) for k in range(2)}
    initial_d_o = {(i, j, k): init_d_o[i, j, k] + 1e-8 for i in range(horizons) for j in range(num_agents) for k in range(num_obs)}
    initial_d_a = {(i, (j, k)): init_d_a[i, j, k] + 1e-8 for i in range(horizons) for j in range(num_agents) for k in range(num_agents) if k > j}


    model.P_out = Var(model.I, model.J, model.Dim, bounds=(-1, 1), initialize=initial_p_out)
    model.d_o = Var(model.I, model.J, model.K, within=NonNegativeReals, initialize=initial_d_o)
    model.d_a = Var(model.I, model.AgentPairs, within=NonNegativeReals, initialize=initial_d_a)


    # Parameters (converted from arrays to dictionaries for Pyomo)
    agent_rads_dict = {j: agent_rads[j] for j in range(num_agents)}
    model.agent_rads = Param(model.J, initialize=agent_rads_dict)

    max_speeds_dict = {j: max_speeds[j] for j in range(num_agents)}
    model.max_speeds = Param(model.J, initialize=max_speeds_dict)

    obs_rads_dict = {k: obs_rads[k] for k in range(num_obs)}
    model.obs_rads = Param(model.K, initialize=obs_rads_dict)

    obs_pos_dict_x = {k: obs_pos[k, 0] for k in range(num_obs)}
    obs_pos_dict_y = {k: obs_pos[k, 1] for k in range(num_obs)}
    model.obs_pos_x = Param(model.K, initialize=obs_pos_dict_x)
    model.obs_pos_y = Param(model.K, initialize=obs_pos_dict_y)

    # Lagrangian multipliers
    nu_o_dict = {(i, j, k): nu_o[i, j, k] for i in range(horizons) for j in range(num_agents) for k in range(num_obs)}
    model.nu_o = Param(model.I, model.J, model.K, initialize=nu_o_dict)

    nu_a_dict = {(i, j, k): nu_a[i, j, k] for i in range(horizons) for j in range(num_agents) for k in range(num_agents) if k > j}
    model.nu_a = Param(model.I, model.AgentPairs, initialize=nu_a_dict)

    # Objective function
    def obj_expression(model):
        expr = 0.0
        # Minimize the distance between predicted path and the true path
        for i in model.I:
            for j in model.J:
                expr += sum((model.P_out[i, j, k] - x[i, j, k]) ** 2 for k in model.Dim)
        # minimize total distance
        for i in model.I:
            for j in model.J:
                if i < model.I.last(): 
                    expr += sum((model.P_out[i + 1, j, k] - model.P_out[i, j, k]) ** 2 for k in model.Dim)
        # Lagrangian terms for obstacle avoidance
        for i in model.I:
            for j in model.J:
                for k in model.K:
                    temp_expr = -((model.P_out[i, j, 0] - model.obs_pos_x[k]) ** 2 + (model.P_out[i, j, 1] - model.obs_pos_y[k]) ** 2) + ((model.agent_rads[j] + model.obs_rads[k])) ** 2 + model.d_o[i, j, k]
                    expr += model.nu_o[i, j, k] * temp_expr
        # Lagrangian terms for collision avoidance
        for i in model.I:
            for j, k in model.AgentPairs:
                temp_expr = -((model.P_out[i, j, 0] - model.P_out[i, k, 0]) ** 2 + (model.P_out[i, j, 1] - model.P_out[i, k, 1]) ** 2) + ((model.agent_rads[j] + model.agent_rads[k])) ** 2 + model.d_a[i, (j, k)]
                expr += model.nu_a[i, (j, k)] * temp_expr
        # Augmented Lagrangian terms for obstacle avoidance
        for i in model.I:
            for j in model.J:
                for k in model.K:
                    temp_expr = -((model.P_out[i, j, 0] - model.obs_pos_x[k]) ** 2 + (model.P_out[i, j, 1] - model.obs_pos_y[k]) ** 2) + ((model.agent_rads[j] + model.obs_rads[k])) ** 2 + model.d_o[i, j, k]
                    expr += rho * temp_expr ** 2
        # Augmented Lagrangian terms for collision avoidance
        for i in model.I:
            for j, k in model.AgentPairs:
                temp_expr = -((model.P_out[i, j, 0] - model.P_out[i, k, 0]) ** 2 + (model.P_out[i, j, 1] - model.P_out[i, k, 1]) ** 2) + ((model.agent_rads[j] + model.agent_rads[k])) ** 2 + model.d_a[i, (j, k)]
                expr += rho * temp_expr ** 2
        return expr

    model.obj = Objective(rule=obj_expression, sense=minimize)

    # Start and end position constraints
    agent_starts_x = {j: agent_starts[j, 0] for j in range(num_agents)}
    agent_starts_y = {j: agent_starts[j, 1] for j in range(num_agents)}
    agent_goals_x = {j: agent_goals[j, 0] for j in range(num_agents)}
    agent_goals_y = {j: agent_goals[j, 1] for j in range(num_agents)}

    def start_position_constraint_x(model, j):
        return model.P_out[0, j, 0] == agent_starts_x[j]

    def start_position_constraint_y(model, j):
        return model.P_out[0, j, 1] == agent_starts_y[j]

    def end_position_constraint_x(model, j):
        return model.P_out[horizons - 1, j, 0] == agent_goals_x[j]

    def end_position_constraint_y(model, j):
        return model.P_out[horizons - 1, j, 1] == agent_goals_y[j]

    model.start_position_constraints_x = Constraint(model.J, rule=start_position_constraint_x)
    model.start_position_constraints_y = Constraint(model.J, rule=start_position_constraint_y)
    model.end_position_constraints_x = Constraint(model.J, rule=end_position_constraint_x)
    model.end_position_constraints_y = Constraint(model.J, rule=end_position_constraint_y)

    # Speed constraints
    def speed_constraint_rule(model, i, j):
        if i < horizons - 1:
            return sum((model.P_out[i, j, k] - model.P_out[i + 1, j, k]) ** 2 for k in model.Dim) <= model.max_speeds[j] ** 2
        else:
            return Constraint.Skip

    model.speed_constraints = Constraint(model.I, model.J, rule=speed_constraint_rule)

    # Solve the model
    solver = SolverFactory('ipopt')
    results = solver.solve(model, tee=False)

    # Extract solution
    x_iteration = np.zeros((horizons, num_agents, 2))
    d_o_value = np.zeros((horizons, num_agents, num_obs))
    d_a_value = np.zeros((horizons, num_agents, num_agents))
    flag = 0
    from pyomo.opt import SolverStatus, TerminationCondition
    if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
        flag = 1
        for i in range(horizons):
            for j in range(num_agents):
                x_iteration[i, j, 0] = value(model.P_out[i, j, 0])
                x_iteration[i, j, 1] = value(model.P_out[i, j, 1])
                for k_obs in range(num_obs):
                    d_o_value[i, j, k_obs] = value(model.d_o[i, j, k_obs])
                for k_agent in range(num_agents):
                    if k_agent > j:
                        d_a_value[i, j, k_agent] = value(model.d_a[i, (j, k_agent)])
    else:
        # Solver failed
        flag = 0
        return x_iteration, d_o_value, d_a_value, flag
    return x_iteration, d_o_value, d_a_value, flag

def grad_nu(agent_rads, obs_pos, obs_rads, p_value, d_o, d_a):
    horizons = p_value.shape[0]
    num_agents = p_value.shape[1]
    num_obs = obs_pos.shape[0] 

    # initialize the gradient of nu_o and nu_a
    grad_nu_o = np.zeros((horizons, num_agents, num_obs))
    grad_nu_a = np.zeros((horizons, num_agents, num_agents))

    # calculate the gradient of nu_o and nu_a
    for i in range(horizons):
        for j in range(num_agents):
            for k in range(num_obs):
                grad_nu_o[i, j, k] = -((p_value[i, j, 0] - obs_pos[k, 0])**2 + (p_value[i, j, 1] - obs_pos[k, 1])**2)+((agent_rads[j] + obs_rads[k]))**2 + d_o[i, j, k]
            for k in range(j+1, num_agents):
                grad_nu_a[i, j, k] = -((p_value[i, j, 0] - p_value[i, k, 0])**2 + (p_value[i, j, 1] - p_value[i, k, 1])**2)+((agent_rads[j] + agent_rads[k]))**2 + d_a[i,j,k]

    return grad_nu_o, grad_nu_a

def check_feasibility(agent_rads, obs_pos, obs_rads, p_value, d_o, d_a, tolerance):
    horizons = p_value.shape[0]
    num_agents = p_value.shape[1]
    num_obs = obs_pos.shape[0] 

    # 

    # initialize the gradient of nu_o and nu_a
    grad_nu_o = np.zeros((horizons, num_agents, num_obs))
    grad_nu_a = np.zeros((horizons, num_agents, num_agents))

    # calculate the gradient of nu_o and nu_a
    for i in range(horizons):
        for j in range(num_agents):
            for k in range(num_obs):
                grad_nu_o[i, j, k] = ((p_value[i, j, 0] - obs_pos[k, 0])**2 + (p_value[i, j, 1] - obs_pos[k, 1])**2)-((agent_rads[j] + obs_rads[k]))**2 
            for k in range(j+1, num_agents):
                grad_nu_a[i, j, k] = ((p_value[i, j, 0] - p_value[i, k, 0])**2 + (p_value[i, j, 1] - p_value[i, k, 1])**2)-((agent_rads[j] + agent_rads[k]))**2 

    # threshold at zero: 1 if feasible (grad ≥ 0), else 0
    feas_o = (grad_nu_o <= -tolerance).astype(int)
    feas_a = (grad_nu_a <= -tolerance).astype(int)

    return np.sum(feas_o), np.sum(feas_a)


def cal_dummy_var(agent_rads, obs_pos, obs_rads, p_value):
    horizons = p_value.shape[0]
    num_agents = p_value.shape[1]
    num_obs = obs_pos.shape[0] 

    # 

    # initialize the gradient of nu_o and nu_a
    d_o = np.zeros((horizons, num_agents, num_obs))
    d_a = np.zeros((horizons, num_agents, num_agents))

    # calculate the gradient of nu_o and nu_a
    for i in range(horizons):
        for j in range(num_agents):
            for k in range(num_obs):
                d_o[i, j, k] = ((p_value[i, j, 0] - obs_pos[k, 0])**2 + (p_value[i, j, 1] - obs_pos[k, 1])**2)-((agent_rads[j] + obs_rads[k]))**2
            for k in range(j+1, num_agents):
                d_a[i,j,k] = ((p_value[i, j, 0] - p_value[i, k, 0])**2 + (p_value[i, j, 1] - p_value[i, k, 1])**2)-((agent_rads[j] + agent_rads[k]))**2

    return d_o, d_a


def apply_projection_alm(x, projection_info, hard_conds, first_projection, init_traj4proj, proj_params):
    # rebuttal
    grad_nu_o_set = []
    grad_nu_a_set = []

    # Get hard conditions from projection_info
    agents_starts_states_normalized = hard_conds[0][0,:]
    agents_goals_states_normalized = hard_conds[63][0,:]

    # get the number of the agents
    num_agents = int(agents_starts_states_normalized.shape[0]/4)

    # unnormalize the x
    agents_starts_states = projection_info.unnormalize_trajectories(agents_starts_states_normalized)
    agents_goals_states = projection_info.unnormalize_trajectories(agents_goals_states_normalized)
    x = projection_info.unnormalize_trajectories(x)

    # get the position of the agents starts and goals
    agents_starts_pos = []
    agents_goals_pos = []
    for i in range(num_agents):
        agents_starts_pos.append([agents_starts_states[i*2], agents_starts_states[i*2+1]])
        agents_goals_pos.append([agents_goals_states[i*2], agents_goals_states[i*2+1]])
    agents_starts_pos = torch.tensor(agents_starts_pos)
    agents_goals_pos = torch.tensor(agents_goals_pos)
    # get the velocity of the agents starts and goals
    agents_starts_v = torch.zeros((num_agents, 2))
    agents_goals_v = torch.zeros((num_agents, 2))

    # get the radius of the agents
    agents_rads = projection_info.robot.radius*1
    # get an array of the agents' radius
    agents_rads = agents_rads*np.ones((num_agents))

    # get the obstacles
    obj_list = list(projection_info.env.obj_all_list)
    

    obs_pos_1 = obj_list[0].fields[0].centers
    obs_rads_1 = obj_list[0].fields[0].radii
    

    obs_pos_2 = obj_list[1].fields[0].centers
    obs_rads_2 = obj_list[1].fields[0].radii
    

    obs_pos = torch.cat([obs_pos_1, obs_pos_2], dim=0)
    obs_rads = torch.cat([obs_rads_1, obs_rads_2], dim=0)


    num_obs = obs_rads.shape[0]

    horizons = 64
    traj_index = np.arange(64)


    agents_max_speeds = proj_params["agents_max_speeds"]



    x_candidate = x[0,:,:].clone()

    agents_starts_pos = agents_starts_pos.cpu().numpy()
    agents_goals_pos = agents_goals_pos.cpu().numpy()
    obs_pos = obs_pos.cpu().numpy()
    obs_rads = obs_rads.cpu().numpy()
    x_candidate = x_candidate.cpu().numpy()




    # Initialize the lagrange multipliers
    nu_o = np.zeros((horizons, num_agents, num_obs))
    nu_a = np.zeros((horizons, num_agents, num_agents))
    p_init = np.zeros((horizons, num_agents, 2))
    d_o = np.zeros((horizons, num_agents, num_obs))
    d_a = np.zeros((horizons, num_agents, num_agents))
    if not first_projection:
        for i in range(horizons):
            for j in range(num_agents):
                p_init[i, j, 0] = x_candidate[traj_index[i], j*2]      
                p_init[i, j, 1] = x_candidate[traj_index[i], j*2+1]      


    else:
        # use the interpolation between the start and end points
        for i in range(horizons):
            for j in range(num_agents):
                p_init[i, j, :] = agents_starts_pos[j, :] + (agents_goals_pos[j, :] - agents_starts_pos[j, :]) * (i / (horizons-1))

    # change the shape of x_candidate
    x_candidate_alm = np.zeros((horizons, num_agents, 2))
    for i in range(horizons):
        for j in range(num_agents):
            x_candidate_alm[i, j, 0] = x_candidate[traj_index[i], j*2]
            x_candidate_alm[i, j, 1] = x_candidate[traj_index[i], j*2+1]
    # calculate the nu_o and nu_a
    # agent_rads, obs_pos, obs_rads, p_value, d_o, d_a
    init_traj4proj_array = np.stack([init_traj4proj[k] for k in init_traj4proj.keys()], axis=1)
    init_d_o, init_d_a = cal_dummy_var(agents_rads, obs_pos, obs_rads, init_traj4proj_array)


    # Set lagrange parameters
    rho = proj_params["rho"]
    rho_factor = proj_params["rho_factor"]

    flag = 0
    grad_o = []
    grad_a = []
    import time
    t_begin = time.time()
    success = 0
    for steps in range(proj_params["alm_iteration"]):
        tt1 = time.time()
        

        x_iteration, d_o, d_a, _ = solve_L_aug(x_candidate_alm, agents_starts_pos, agents_goals_pos, agents_rads, agents_max_speeds, obs_pos, obs_rads, horizons, nu_o, nu_a, rho, init_traj4proj_array, init_d_o, init_d_a)
        grad_nu_o, grad_nu_a = grad_nu(agents_rads, obs_pos, obs_rads, x_iteration, d_o, d_a)

        init_traj4proj_array = x_iteration
        init_d_o, init_d_a = d_o, d_a


        fea_o_real, fea_a_real = check_feasibility(agents_rads, obs_pos, obs_rads, x_iteration, d_o, d_a, proj_params["tolerance"])

        # # norm
        print(steps)
        print('grad for solving alm')
        print(np.mean(np.linalg.norm(grad_nu_o)))
        print(np.mean(np.linalg.norm(grad_nu_a)))
        print('grad for real settings')
        print(np.mean(fea_o_real))
        print(np.mean(fea_a_real))


        grad_nu_o_set.append(np.mean(np.linalg.norm(grad_nu_o)))
        grad_nu_a_set.append(np.mean(np.linalg.norm(grad_nu_a)))
        if (np.mean(np.linalg.norm(grad_nu_o)) <= 1e-3 and np.mean(np.linalg.norm(grad_nu_a)) <= 1e-3) or (fea_o_real+fea_a_real<1):
            success = 1
            break

        else:
            # update the lagrange parameters
            rho = rho * rho_factor
            nu_o = nu_o + rho * grad_nu_o
            nu_a = nu_a + rho * grad_nu_a
            grad_o.append(np.mean(np.linalg.norm(grad_nu_o)))
            grad_a.append(np.mean(np.linalg.norm(grad_nu_a)))
        
        tt2 = time.time()
        print('Time for each iteration:', tt2 - tt1)
        
        
    t_end = time.time()
    print('Time:', t_end - t_begin)


  
    x_projected = copy.copy(x_candidate)
    for i in range(horizons):
        for j in range(num_agents):
            x_projected[traj_index[i], j*2] = x_iteration[i, j, 0]    
            x_projected[traj_index[i], j*2+1] = x_iteration[i, j, 1]
                

    for i in range(0, x.shape[0]):
        x_projected = torch.tensor(x_projected).to(x.dtype).to(x.device)
        x_projected = projection_info.normalize_trajectories(x_projected)
        x[i,:,:] = x_projected


    return x, success

