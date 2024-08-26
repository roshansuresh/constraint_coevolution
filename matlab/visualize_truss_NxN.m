function [] = visualize_truss_NxN(NC,CA,sidenum,add_labels)
% This function provides a graphic visualization of the NxN truss 
% represented by CA

% Plot node positions if needed
if add_labels
    labels = cell(sidenum^2,1);
    for j = 1:sidenum^2
        labels{j} = num2str(j);
    end
    for i = 1:size(NC,1)
        plot(NC(i,1),NC(i,2),'*r')
        hold on
        text(NC(i,1),NC(i,2),labels{i},'HorizontalAlignment','left','FontSize',15,'Color','m','FontWeight','bold')
        hold on
    end
end

% Plot truss elements one-by-one
for i = 1:size(CA,1)
    % Finding Positions of truss element end points
    x1 = NC(CA(i,1),1); 
    y1 = NC(CA(i,1),2);
    x2 = NC(CA(i,2),1);
    y2 = NC(CA(i,2),2);
    % Plotting line between the two end points
    %x_val = linspace(x1,x2);
    %y_val = linspace(y1,y2);
    plot([x1,x2],[y1,y2],'-b','LineWidth',2)
    hold on
end
hold off

end