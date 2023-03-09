% This file contains functions for evaluating models for the 2023 Challenge. You can run it as follows:
%
%   evaluate_model(labels, outputs, results)
%
% where 'labels' is a folder containing files with the labels, 'outputs' is a folder containing files with the outputs from your
% model, and 'results' (optional) is the folder to store the collection of scores for the model outputs.
%
% Each label or output file must have the format described on the Challenge webpage. The scores for the algorithm outputs include
% the area under the receiver-operating characteristic curve (AUROC), the area under the recall-precision curve (AUPRC), macro
% accuracy, a weighted accuracy score, and the Challenge score.

function evaluate_model(labels, outputs, results)
    % Check for Python and NumPy.
    command = 'python -V';
    [status, ~] = system(command);
    if status~=0
        error('Python not found: please install Python or make it available by running "python ...".');
    end

    command = 'python -c "import numpy"';
    [status, ~] = system(command);
    if status~=0
        error('NumPy not found: please install NumPy or make it available to Python.');
    end

    % Define command for evaluating model outputs.
    switch nargin
        case 2
            command = ['python evaluate_model.py' ' ' labels ' ' outputs];
        case 3
            command = ['python evaluate_model.py' ' ' labels ' ' outputs ' ' results];
        otherwise
            command = '';
    end

    % Evaluate model outputs.
    [~, output] = system(command);
    fprintf(output);
end