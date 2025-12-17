%Classification using perceptron

%Reading apple images
A1=imread('apple_04.jpg');
A2=imread('apple_05.jpg');
A3=imread('apple_06.jpg');
A4=imread('apple_07.jpg');
A5=imread('apple_11.jpg');
A6=imread('apple_12.jpg');
A7=imread('apple_13.jpg');
A8=imread('apple_17.jpg');
A9=imread('apple_19.jpg');

%Reading pears images
P1=imread('pear_01.jpg');
P2=imread('pear_02.jpg');
P3=imread('pear_03.jpg');
P4=imread('pear_09.jpg');

%Calculate for each image, colour and roundness
%For Apples
%1st apple image(A1)
hsv_value_A1=spalva_color(A1); %color
metric_A1=apvalumas_roundness(A1); %roundness
%2nd apple image(A2)
hsv_value_A2=spalva_color(A2); %color
metric_A2=apvalumas_roundness(A2); %roundness
%3rd apple image(A3)
hsv_value_A3=spalva_color(A3); %color
metric_A3=apvalumas_roundness(A3); %roundness
%4th apple image(A4)
hsv_value_A4=spalva_color(A4); %color
metric_A4=apvalumas_roundness(A4); %roundness
%5th apple image(A5)
hsv_value_A5=spalva_color(A5); %color
metric_A5=apvalumas_roundness(A5); %roundness
%6th apple image(A6)
hsv_value_A6=spalva_color(A6); %color
metric_A6=apvalumas_roundness(A6); %roundness
%7th apple image(A7)
hsv_value_A7=spalva_color(A7); %color
metric_A7=apvalumas_roundness(A7); %roundness
%8th apple image(A8)
hsv_value_A8=spalva_color(A8); %color
metric_A8=apvalumas_roundness(A8); %roundness
%9th apple image(A9)
hsv_value_A9=spalva_color(A9); %color
metric_A9=apvalumas_roundness(A9); %roundness

%For Pears
%1st pear image(P1)
hsv_value_P1=spalva_color(P1); %color
metric_P1=apvalumas_roundness(P1); %roundness
%2nd pear image(P2)
hsv_value_P2=spalva_color(P2); %color
metric_P2=apvalumas_roundness(P2); %roundness
%3rd pear image(P3)
hsv_value_P3=spalva_color(P3); %color
metric_P3=apvalumas_roundness(P3); %roundness
%2nd pear image(P4)
hsv_value_P4=spalva_color(P4); %color
metric_P4=apvalumas_roundness(P4); %roundness

%selecting features(color, roundness, 3 apples and 2 pears)
%A1,A2,A3,P1,P2
%building matrix 2x5
x1Training=[hsv_value_A1 hsv_value_A2 hsv_value_A3 hsv_value_P1 hsv_value_P2];
x2Training=[metric_A1 metric_A2 metric_A3 metric_P1 metric_P2];

%--------------------------------------------
x1Ats=[hsv_value_A4 hsv_value_A5 hsv_value_A6 hsv_value_A7 hsv_value_A8 hsv_value_A9 hsv_value_P3 hsv_value_P4];
x2Ats=[metric_A4 metric_A5 metric_A6 metric_A7 metric_A8 metric_A9 metric_P3 metric_P4];
%--------------------------------------------

%--------------------------------------------
% estimated features are stored in matrix P:
P=[x1Training;x2Training];
%--------------------------------------------

%--------------------------------------------
%Desired output vector. Apples: 1 and Pears: -1
T=[1;1;1;-1;-1]; %Had to fix values to match training data
%--------------------------------------------

%% train single perceptron with two inputs and one output

%--------------------------------------------
% Sample amount
N=length(x1Training);
%--------------------------------------------

% generate random initial values of w1, w2 and b
w1 = randn(1);
w2 = randn(1);
b = randn(1);

%--------------------------------------------
learningRate = 0.1;

%Required to start and enter the loop where error value is recalculated
totalError = 1; 
%--------------------------------------------


%--------------------------------------------
% write training algorithm
while totalError ~= 0 % executes while the total error is not 0
	% here should be your code of parameter update
    totalError = 0;

    for n = 1:N
        
        %Current input for example n
        x1color = x1Training(n);
        x2roundness = x2Training(n);  
        
        %Weighted sum
        v = x1color*w1 + x2roundness*w2 + b;

        %Perceptron output
        if v > 0
            y = 1;
        else
            y = -1;
        end

        %Error calculation
        e = T(n) - y;

        % Update weights and bias
        w1 = w1 + learningRate * e * x1color;
        w2 = w2 + learningRate * e * x2roundness;
        b = b + learningRate * e;

        % Calculate the total error for all inputs
        totalError = totalError + abs(e);
    end

	% calculate the total error for these 5 inputs 
	%e = abs(e1) + abs(e2) + abs(e3) + abs(e4) + abs(e5);
    

end 
fprintf('Training finished.\n');
fprintf('w1 = %f, w2 = %f, b = %f\n', w1, w2, b);

% Test on additional examples (A4..A9, P3, P4)
N_test = length(x1Ats);
Y_test = zeros(N_test,1);

for n = 1:N_test
    v = x1Ats(n)*w1 + x2Ats(n)*w2 + b;
    if v > 0
        Y_test(n) = 1;
    else
        Y_test(n) = -1;
    end
end

fprintf("\nPerceptron predictions for test images:\n");
fprintf('"%d" ', Y_test);


%Bayes
% Training data matrix (5 samples, each sample = [color roundness])
PBay = [x1Training; x2Training]';     % transpose → 5×2 matrix

% Train Naive Bayes model
BayesModel = fitcnb(PBay, T);

% Testing data matrix (8 samples: A4..A9, P3, P4)
PBayAts = [x1Ats; x2Ats]';    % 8×2 matrix

% Predict classes
BayesAts = predict(BayesModel, PBayAts);

% Print results
fprintf('\n------Bayes results------\n');
fprintf('"%d" ', BayesAts);
fprintf('\n');
%--------------------------------------------

%For result check
%--------------------------------------------
figure; hold on;
scatter(x1Training(1:3), x2Training(1:3), 120, 'r', 'filled');  % apples
 scatter(x1Training(4:5), x2Training(4:5), 120, 'b', 'filled');  % pears
 xlabel('Color feature');
 ylabel('Roundness feature');
 title('Training Data');
 legend('Apples','Pears');
 grid on;
%--------------------------------------------