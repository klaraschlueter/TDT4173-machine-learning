module XorNN(xorSet, testRun, trainPlot) where

import SimpleNN

import Numeric.LinearAlgebra.HMatrix
import Numeric.LinearAlgebra.Data
import Graphics.EasyPlot
import Debug.Trace
import Definitions


-- | The data set containing input and target output for the XOR problem.
xorSet :: [(Matrix Double, Matrix Double)]

xorSet = [  (matrix 2 [0, 0], ma0), (matrix 2 [0, 1], ma1),
            (matrix 2 [1, 0], ma1), (matrix 2 [1, 1], ma0)]
    where   ma0 = matrix 1 [0]
            ma1 = matrix 1 [1]



-- | Trains a small neural network (architecture as shown in the report) for the XOR problem.
-- Uses the trained network to predict the output for every instance of the XOR problem and
-- displays the results. The learning rate is set to 1, and the training consists of 2000 epochs.
testRun :: IO ()

testRun = do
    weights <- nnInit [2, 2, 1]
    let trainedWeights = nnTrain weights xorSet learningRate numberEpochs
        predictions = map round $ map (xorNNRun trainedWeights) (map fst xorSet)
        beautifulOutput = "predictions of trained network (learning rate " ++ (show learningRate)
                        ++ ", epochs " ++ (show numberEpochs) ++ ") for the instances of the XOR problem:"
                        ++ " \n " ++ "(0, 0) ~> " ++ (show $ predictions !! 0)
                        ++ " \n " ++ "(0, 1) ~> " ++ (show $ predictions !! 1)
                        ++ " \n " ++ "(1, 0) ~> " ++ (show $ predictions !! 2)
                        ++ " \n " ++ "(1, 1) ~> " ++ (show $ predictions !! 3)
        learningRate = 1
        numberEpochs = 2000
    putStrLn beautifulOutput



-- | Plots the training of a small network (architecture as shown in the report) for the XOR
-- problem. The learning rate is set to 1, and the training consists of 2000 epochs.
trainPlot :: IO Bool

trainPlot = do
        initWs <- nnInit [2, 2, 1]
        let allLosses = losses initWs numberEpochs

            loss00 = zipWith (\x y -> (x,y)) [1..numberEpochs] (map (flip (!!) 0) allLosses)
            loss01 = zipWith (\x y -> (x,y)) [1..numberEpochs] (map (flip (!!) 1) allLosses)
            loss10 = zipWith (\x y -> (x,y)) [1..numberEpochs] (map (flip (!!) 2) allLosses)
            loss11 = zipWith (\x y -> (x,y)) [1..numberEpochs] (map (flip (!!) 3) allLosses)           

        plot X11 [Data2D [Title "error for input (0, 0)", Style Points] [] loss00,    
                  Data2D [Title "error for input (0, 1)", Style Points] [] loss01,
                  Data2D [Title "error for input (1, 0)", Style Points] [] loss10,
                  Data2D [Title "error for input (1, 1)", Style Points] [] loss11]

    where losses :: [Matrix Double] -> Double -> [[Double]]
          losses ws 0 = []
          losses ws n = let wsAfterEpoch = epoch ws xorSet learningRate
                            currentLoss = setLoss ws
                            succLosses = losses wsAfterEpoch (n - 1)
                        in (currentLoss : succLosses)

          learningRate = 1
          numberEpochs = 2000 



-- Helper functions -------------------------------------------------------------------------

setLoss :: [Matrix Double] -> [Double]
setLoss ws = zipWith loss (map (xorNNRun ws) $ map fst xorSet) (map (singleValue.snd) xorSet) 

xorNNRun :: [Matrix Double] -> Matrix Double -> Double
xorNNRun ws input = singleValue $ nnPredict input ws

singleValue :: Matrix Double -> Double
singleValue m = ((toList.flatten) m) !! 0
