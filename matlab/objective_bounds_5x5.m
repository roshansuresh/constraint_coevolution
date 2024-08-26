%% Find max and min bounds of the objectives for 5x5 nodegrid (only random designs or epsilon MOEA runs)
clear
close all
clc

%% Problem parameters (printable designs)
prob_truss = true; % if true -> truss problem, if false -> artery problem

E = 1.8162e6; % Young's Modulus for polymeric material (example: 1.8162 MPa for SIL material)
sel = 10e-3; % Unit square side length (NOT individual truss length) (in m)
r = 250e-6; % Radius for cross-sectional area of (assumed circular) truss members (in m)
A = pi*(r^2); % Cross-sectional area of truss member
sidenum = 5;
biasFactor = 1;
collapsibilityBiasFac = 0.5;
choice_of_model = "Truss"; % "Fibre" -> fibre model, "Truss" -> truss model, "Beam" -> beam model

n_members_total = nchoosek(sidenum^2,2); 

c_ratio = 0.421;
if prob_truss
    c_ratio = 1;
end

CA_all = get_CA_all(sidenum);
NC = generateNC(sel, sidenum);

n_members_repeated = 2*nchoosek(sidenum,2);
n_variables = n_members_total - n_members_repeated;

%% Generate and evaluate randomly generated designs
rand_only = false; % whether to evaluate only randomly generated designs or read results from epsilon MOEA runs

num_runs = 10;
c11_bounds_allruns = zeros(num_runs,2); % bounds for each run -> [max, min]
c22_bounds_allruns = zeros(num_runs,2); % bounds for each run -> [max, min]
volfrac_bounds_allruns = zeros(num_runs,2); % bounds for each run -> [max, min]
c11volfrac_bounds_allruns = zeros(num_runs,2); % bounds for each run -> [max, min]
c12c11_bounds_allruns = zeros(num_runs,2); % bounds for each run -> [max, min]
c21c11_bounds_allruns = zeros(num_runs,2); % bounds for each run -> [max, min]
c22c11_1_bounds_allruns = zeros(num_runs,2); % bounds for each run -> [max, min]
c22c11_c_rat_bounds_allruns = zeros(num_runs,2); % bounds for each run -> [max, min]
c61_bounds_allruns = zeros(num_runs,2); % bounds for each run -> [max, min]
c62_bounds_allruns = zeros(num_runs,2); % bounds for each run -> [max, min]
c16_bounds_allruns = zeros(num_runs,2); % bounds for each run -> [max, min]
c26_bounds_allruns = zeros(num_runs,2); % bounds for each run -> [max, min]
c66c11_bounds_allruns = zeros(num_runs,2); % bounds for each run -> [max, min]

if rand_only
    n_des = 1000;
end

for i = 1:num_runs
    if rand_only
        [design_array_run, ~, ~, ~, ~] = generate_biased_random_population(n_des, choice_of_model, prob_truss, "Random", sidenum, sel, r, E, c_ratio, collapsibilityBiasFac, biasFactor);
    else
        [~, design_array_run] = read_csv_data(prob_truss, choice_of_model, n_variables, i-1);
    end
    
    n_des = size(design_array_run,2);
    c11_vals_run = zeros(n_des, 1);
    c22_vals_run = zeros(n_des, 1);
    volfrac_vals_run = zeros(n_des, 1);
    c11volfrac_vals_run = zeros(n_des, 1);
    c22c11_1_vals_run = zeros(n_des, 1);
    c22c11_c_rat_vals_run = zeros(n_des, 1);
    c12c11_vals_run = zeros(n_des, 1);
    c21c11_vals_run = zeros(n_des, 1);
    c61_vals_run = zeros(n_des, 1);
    c62_vals_run = zeros(n_des, 1);
    c16_vals_run = zeros(n_des, 1);
    c26_vals_run = zeros(n_des, 1);
    c66c11_vals_run = zeros(n_des, 1);
    des_count = 1;

    for j = 1:n_des
        if rand_only
            x_des_current = design_array_run(:,j); 
            CA_des = CA_all(x_des_current == 1,:);
        else
            x_bool_current = zeros(size(design_array_run,2),1);
            for k = 1:size(design_array_run,2)
                if strcmp(design_array_run{j,k},'true')
                    x_bool_current(k,1) = true;
                else
                    x_bool_current(k,1) = false;
                end
            end
            x_complete_des = get_complete_boolean_array(x_bool_current, sidenum);   
            CA_des = CA_all(x_complete_des == 1,:);
        end
        rvar_des = r.*ones(1,size(CA_des,1));
        switch choice_of_model
            case "Fibre"
                if truss_problem
                    [C11,C22,volfrac_des] = fiberStiffnessModel_rVar_V3(sel,rvar_des,E,CA_des,nucFac,sidenum);
                    C_des = zeros(6);
                    C_des(1,1) = C11;
                    C_des(2,2) = C22;
                else
                    disp("Fiber stiffness model not suitable for artery problem")
                    exit
                end
            case "Truss"
                [C_des, ~] = trussMetaCalc_NxN_1UC_rVar_AVar(sidenum,sel,rvar_des,E,CA_des);
                volfrac_des = calcVF_NxN_feasOnly(CA_des,r,sel,sidenum);
            case "Beam"
                C_des = Beam_2D_NxN_PBC(sel,sidenum,r,E,CA_des);
                volfrac_des = calcVF_NxN_feasOnly(CA_des,r,sel,sidenum);
        end
        if (any(C_des(1:2,1:2) < 1,'all') || any(C_des > E,'all') || any(isnan(C_des),'all'))
            continue
        end
        c11_vals_run(des_count,1) = C_des(1,1);
        c22_vals_run(des_count,1) = C_des(2,2);
        volfrac_vals_run(des_count,1) = volfrac_des;
        c11volfrac_vals_run(des_count,1) = C_des(1,1)/volfrac_des;
        c22c11_1_vals_run(des_count,1) = abs((C_des(2,2)/C_des(1,1)) - 1);
        c22c11_c_rat_vals_run(des_count,1) = abs((C_des(2,2)/C_des(1,1)) - 0.421);
        c12c11_vals_run(des_count,1) = abs((C_des(1,2)/C_des(1,1)) - 0.0745);
        c21c11_vals_run(des_count,1) = abs((C_des(2,1)/C_des(1,1)) - 0.0745);
        c61_vals_run(des_count,1) = abs(C_des(3,1));
        c62_vals_run(des_count,1) = abs(C_des(3,2));
        c16_vals_run(des_count,1) = abs(C_des(1,3));
        c26_vals_run(des_count,1) = abs(C_des(2,3));
        c66c11_vals_run(des_count,1) = abs((C_des(3,3)/C_des(1,1)) - 5.038);

        des_count = des_count + 1;
    end

    c11_bounds_allruns(i,:) = [max(c11_vals_run(1:des_count-1)), min(c11_vals_run(1:des_count-1))];
    c22_bounds_allruns(i,:) = [max(c22_vals_run(1:des_count-1)), min(c22_vals_run(1:des_count-1))];
    volfrac_bounds_allruns(i,:) = [max(volfrac_vals_run(1:des_count-1)), min(volfrac_vals_run(1:des_count-1))];
    c11volfrac_bounds_allruns(i,:) = [max(c11volfrac_vals_run(1:des_count-1)), min(c11volfrac_vals_run(1:des_count-1))];
    c12c11_bounds_allruns(i,:) = [max(c12c11_vals_run(1:des_count-1)), min(c12c11_vals_run(1:des_count-1))];
    c21c11_bounds_allruns(i,:) = [max(c21c11_vals_run(1:des_count-1)), min(c21c11_vals_run(1:des_count-1))];
    c22c11_1_bounds_allruns(i,:) = [max(c22c11_1_vals_run(1:des_count-1)), min(c22c11_1_vals_run(1:des_count-1))];
    c22c11_c_rat_bounds_allruns(i,:) = [max(c22c11_c_rat_vals_run(1:des_count-1)), min(c22c11_c_rat_vals_run(1:des_count-1))];
    c61_bounds_allruns(i,:) = [max(c61_vals_run(1:des_count-1)), min(c61_vals_run(1:des_count-1))];
    c62_bounds_allruns(i,:) = [max(c62_vals_run(1:des_count-1)), min(c62_vals_run(1:des_count-1))];
    c16_bounds_allruns(i,:) = [max(c16_vals_run(1:des_count-1)), min(c16_vals_run(1:des_count-1))];
    c26_bounds_allruns(i,:) = [max(c26_vals_run(1:des_count-1)), min(c26_vals_run(1:des_count-1))];
    c66c11_bounds_allruns(i,:) = [max(c66c11_vals_run(1:des_count-1)), min(c66c11_vals_run(1:des_count-1))];
end

%% Functions
function [data_array, design_array] = read_csv_data(problem_truss, choice_of_model, num_var, run_num)
    filepath = "C:\\SEAK Lab\\SEAK Lab Github\\Heuristics in RL\\results\\";
    
    if problem_truss
        filepath_prob = "Equal Stiffness\\";
    else
        filepath_prob = "Artery\\";
    end
    
    filename = "EpsilonMOEA_";
    filepath_moea = "Epsilon MOEA\\";
    
    filename2 = "_allSolutions.csv";
    
    switch choice_of_model
        case "Fibre"
            filepath3 = "Fibre Model\\";
            if ~truss_problem              
                disp("Fiber stiffness model not suitable for artery problem")
                exit
            end
        case "Truss"
            filepath3 = "Truss Model\\";
        case "Beam"
            filepath3 = "Beam Model\\";
    end
    
    %%%% read appropriate file 
    full_filepath = strcat(filepath,filepath_prob,filepath_moea,filepath3,"original 3x3 bounds - old feas\\",filename,num2str(run_num),filename2);
    
    if problem_truss
        n_data = num_var + 1 + 2 + 3 + 4; % number of variables + 1 (NFE) + number of objectives + number of constraints + number of heuristics
    else
        n_data = num_var + 1 + 2 + 2 + 4;
    end
	
    %format_string = '%s';
    %for j = 1:n_data
        %format_string = strcat(format_string,'%f');
    %end
    data_table = readtable(full_filepath,'ReadVariableNames',false);
    
    %%%% store retrieved data into different variables
    %%%% for the truss problem:
    %%%% csv_data includes: [NFE, Pen. Obj. 1, Pen.Obj. 2, True Obj. 1, True Obj. 2, Feasibility Score,
    %%%% Connectivity Score, Stiffness Ratio Constraint, Partial Collapsibility Score, 
    %%%% Nodal Properties Score, Orientation Score]
    %%%% for the artery problem:
    %%%% csv_data includes: [NFE, Pen. Obj. 1, Pen.Obj. 2, True Obj. 1, True Obj. 2, Feasibility Score,
    %%%% Connectivity Score, Partial Collapsibility Score, Nodal Properties Score, Orientation Score]
    
	pop_size =  size(data_table,1);
    
    designs = data_table(:,2:num_var+1);
    csv_data = data_table(:,num_var+2:end);
    
    data_array = table2array(csv_data);
    design_cell = table2array(designs);
    
    design_array = strings(pop_size, size(design_cell, 2));
    for j = 1:pop_size
        for k = 1:size(design_cell,2)
            design_array(j,k) = design_cell{j,k};
        end
    end
    
end

function x_des = get_binary_array_from_bitstring(des_string)
	x_des = zeros(strlength(des_string),1);
	for i = 1:strlength(des_string)
		x_des(i,1) = str2double(des_string(i));
	end
end