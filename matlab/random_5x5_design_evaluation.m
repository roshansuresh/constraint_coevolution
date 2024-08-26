%% Evaluation of random 5x5 design to validate models
clear
close all
clc

%% Problem parameters (printable designs)
prob_truss = false; % if true -> truss problem, if false -> artery problem

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

%% Generate random design and evaluate
n_des = 1;
[design_array_run, ~, ~, ~, ~] = generate_biased_random_population(n_des, choice_of_model, prob_truss, "Random", sidenum, sel, r, E, c_ratio, collapsibilityBiasFac, biasFactor);
CA_des = CA_all(design_array_run == 1,:);
rvar_des = r.*ones(1,size(CA_des,1));

% Visualize design
visualize_truss_NxN(NC, CA_des, sidenum, false);

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
des_str = get_design_string(design_array_run);

% Evaluate constraints
feas_des = feasibility_checker_nonbinary_V5(NC,CA_des,sel,sidenum);
conn_des = connectivityConstraint_PBC_2D(sidenum,NC,CA_des,sel,biasFactor);
if strcmp(choice_of_model,"Truss")
    stiffrat_des = abs((C_des(2,2)/C_des(1,1)) - c_ratio);
end

% Evaluate heuristics
partcoll_des = partCollapseHeuristic_2D(sidenum,CA_des,NC,sel,biasFactor);
nodalprop_des = connectivityHeuristic_2D(sidenum,NC,CA_des,sel,biasFactor);
orient_des = orientationHeuristic_V2(NC,CA_des,c_ratio);
inters_des = intersectHeuristic(NC,CA_des);


%% Functions
function des_str = get_design_string(design_array)
    des_str = '';
    for i = 1:size(design_array,1)
        des_str = strcat(des_str,num2str(design_array(i)));
    end
end
