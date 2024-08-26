%% Evaluate specific design (to check consistency between java and python)
clear
clc 

%% Problem parameters
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

%% Evaluate design
%x_des = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];

x_des_str = '0110111011100001100011111101010110001011001110010110111110010111100000110100000101001001100110111000010110111110100010011000010011111001110010010010110100100111011011111001111110100000101111100101101000110010000011010111101011110011111111000100010111101100001101111011001001110000';
x_des = get_binary_array_from_bitstring(x_des_str);

x_complete_des = get_complete_boolean_array(x_des, sidenum);           
CA_des = CA_all(x_complete_des~=0,:);
rvar_des = r.*ones(1,size(CA_des,1));

% Visualize design
visualize_truss_NxN(NC, CA_des, sidenum, true);

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

%% Designs comparison
x_des_str1 = '0110111011100001100011111101010110001011101110010110111110010111100000110100000101001001100110111010010110111110100010011000010011111001110010010010110100100111011011111001111110100000101111100101101000110010000011010111101011110011111111000100010111101100001101111011001001110000';
x_des1 = get_binary_array_from_bitstring(x_des_str1);

x_des_str2 = '0110111011100001100011111101010110001011101100010110111110010111100000110100000101001001100110111010010110111110100010011000010011111001110010010010110100100111011011111001111110100000101111100101101000110010000011010111101011110011111111000100010111101100001101111011001001110000';
x_des2 = get_binary_array_from_bitstring(x_des_str2);
%% Functions

function x_des = get_binary_array_from_bitstring(des_string)
	x_des = zeros(strlength(des_string),1);
	for i = 1:strlength(des_string)
		x_des(i,1) = str2double(des_string(i));
	end
end
