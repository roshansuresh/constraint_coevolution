%% Find objective bounds for each C-DTLZ problem suing Epsilon-MOEA runs
clear
clc

%% Main
problem_choice = 2; 
% 1 -> C1-DTLZ1
% 2 -> C1-DTLZ3
% 3 -> C2-DTLZ2
% 4 -> C3-DTLZ1
% 5 -> C3-DTLZ4

n_objs = 3; % choices -> {3,6,12}

n_constrs = 1;
if problem_choice == 4 || problem_choice == 5
    n_constrs = n_objs;
end

num_runs = 30;

[obj_bounds, constr_bounds] = read_and_get_bounds(problem_choice, n_objs, n_constrs, num_runs); 

%% Functions
function [objs_bounds, constrs_bounds] = read_and_get_bounds(prob_choice, n_objs, n_constrs, n_runs)
    filepath = "C:\\SEAK Lab\\Coev Constr results\\";
    filename = "EpsilonMOEA";
    filepath_moea = "Epsilon MOEA\\";
    
    problem_path = "C1-DTLZ1\\";
    problem_filename = "_c1_dtlz1";
    if prob_choice == 2
        problem_path = "C1-DTLZ3\\";
        problem_filename = "_c1_dtlz3";
    elseif prob_choice == 3
        problem_path = "C2-DTLZ2\\";
        problem_filename = "_c2_dtlz2";
    elseif prob_choice == 4
        problem_path = "C3-DTLZ1\\";
        problem_filename = "_c3_dtlz1";
    elseif prob_choice == 5
        problem_path = "C3-DTLZ4\\";
        problem_filename = "_c3_dtlz4";
    end
    
    filepath_internal = "for norms calculation\\";
    
    objs_filepath = strcat(num2str(n_objs)," objectives\\");
    objs_filename = strcat("_",num2str(n_objs));
    
    objs_bounds_runs = zeros(n_objs,2,n_runs);
    constrs_bounds_runs = zeros(n_constrs,2,n_runs);
    for i = 1:n_runs
        filename_run = strcat("_",num2str(i-1),"_allsolutions.csv");
        full_filename = strcat(filename,problem_filename,objs_filename,filename_run);
        
        full_filepath = strcat(filepath,problem_path,objs_filepath,filepath_moea,filepath_internal,full_filename);
        
        format_string = "";
        if prob_choice == 1 
            num_vars = n_objs + 5 - 1; % k = 5
        elseif prob_choice == 2 || prob_choice == 3 
            num_vars = n_objs + 10 - 1;  % k = 10
        elseif prob_choice == 4
            num_vars = (2*n_objs) + 5; % k = 5
        else 
            num_vars = (2*n_objs) + 10; % k = 10
        end
            
        for j = 1:(num_vars + 1 + n_objs + 1) % num_vars + NFE + n_objs + constraint
            format_string = strcat(format_string,"%f");
        end
        
        data_table = readtable(full_filepath,'Format',format_string,'Delimiter',',');
        
        for j = 1:n_objs
            col_name = strcat("TrueObjective",num2str(j));
            obj_vals = data_table.(col_name);
            
            objs_bounds_runs(j,1,i) = min(obj_vals);
            objs_bounds_runs(j,2,i) = max(obj_vals);
        end
        
        for j = 1:n_constrs
            col_name = strcat("Constraint",num2str(j-1));
            constr_vals = data_table.(col_name);
            
            constrs_bounds_runs(j,1,i) = min(constr_vals);
            constrs_bounds_runs(j,2,i) = max(constr_vals);
        end
    end
    
    objs_bounds = zeros(n_objs,2);
    for i = 1:n_objs
        objs_bounds(i,1) = min(objs_bounds_runs(i,1,:));
        objs_bounds(i,2) = max(objs_bounds_runs(i,2,:));
    end
    
    constrs_bounds = zeros(n_constrs,2);
    for i = 1:n_constrs
        constrs_bounds(i,1) = min(constrs_bounds_runs(i,1,:));
        constrs_bounds(i,2) = max(constrs_bounds_runs(i,2,:));
    end
end
