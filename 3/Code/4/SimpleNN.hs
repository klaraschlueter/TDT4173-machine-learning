module SimpleNN where

import Definitions
import Numeric.LinearAlgebra.HMatrix
import System.Random

-- | Returns randomly initialised weights of a neural network as a list of matrices (one matrix for
-- the weights of each layer), wrapped in the IO monad due to the generation of random numbers. The
-- given list determines the number of layers and neurons: per list element, a layer with the
-- corresponding number of neurons is initialised.
nnInit :: [Int]                 -- ^ Determines the number of layers and neurons: per list element,
                                --   a layer with the corresponding number of neurons is initialised.
        -> IO [Matrix Double]   -- ^ Returns the randomly initialised weights of a network with the
                                --   given dimensions.


nnInit (out:[])         = return []
nnInit (from:to:layers) = do
    gen <- newStdGen
    recursiveWeights <- nnInit (to:layers)
    let randoms = take (from * to) $ randomRs (0, 1) gen
        weights = (matrix to randoms)
    return (weights:recursiveWeights)


-- | Returns the weights of the network after training.
nnTrain :: [Matrix Double]              -- ^ The weights of the network.
    -> [(Matrix Double, Matrix Double)] -- ^ The training data as a list of examples. One
                                        --   example consists of an input vector and an expected
                                        --   output vector. Both have to be row vectors.
    -> Double                           -- ^ The learning rate.
    -> Int                              -- ^ The number of epochs.
    -> [Matrix Double]                  -- ^ Returns the weights after training.

nnTrain weights trainSet learnR epochs  = if epochs < 0 then error $ "Number of training epochs must "
                                                                   ++ "be positive!"
                                          else train weights epochs

    where   train weights 0 = weights
            train weights n = train (epoch weights trainSet learnR) (n-1)


-- | Returns the new weights after one epoch of training with respect to the given training data.
epoch :: [Matrix Double]                -- ^ The weights of the network.
    -> [(Matrix Double, Matrix Double)] -- ^ The training data as a list of examples. One
                                        --   example consists of an input vector and an expected
                                        --   output vector. Both have to be row vectors.
    -> Double                           -- ^ The learning rate.
    -> [Matrix Double]                  -- ^ Returns the updated weights of the network.

epoch weights trainSet learnR = zipWith (-) weights (map (*learningRate) gradientDescent)

    where gradientDescent = gradDesc weights trainSet
          learningRate = matrix 1 [learnR]


-- | Returns the gradient descent of the network represented by the given weights, calculated for
-- for the given set of training data.
gradDesc :: [Matrix Double]             -- ^ The weights of the network.
    -> [(Matrix Double, Matrix Double)] -- ^ The training data as a list of examples. One
                                        --   example consists of an input vector and an expected
                                        --   output vector. Both have to be row vectors.
    -> [Matrix Double]                  -- ^ Returns the gradient descent for the training set.

gradDesc weights trainSet = let gradDescList = (map (singleGradDesc weights) trainSet) :: [[Matrix Double]]
                            in foldl1 (zipWith (+)) gradDescList :: [Matrix Double]

 
-- | Returns the gradient descent of the network represented by the given weights, calculated for
-- one single training example.
singleGradDesc :: [Matrix Double]           -- ^ The weights of the network.
        -> (Matrix Double, Matrix Double)   -- ^ The training example as an input vector and an
                                            --   expected output vector. Both have to be row
                                            --   vectors.
        -> [Matrix Double]                  -- ^ Returns the gradient descent for the training example.

singleGradDesc weights (actIn, tarOut) = snd $ backprop weights actIn
    
    where backprop (w:ws) prevAct
            | ws == []          =   let del = (act - tarOut) * (cmap sigma' z)
                                    in (del, [desc del])

            | otherwise         =   let del = (succDel `mul` (tr succW)) * (cmap sigma' z)
                                        (succW:_) = ws
                                        (succDel, succDescs) = backprop ws act
                                    in  (del, (desc del) : succDescs)

                where act = cmap sigma z
                      z = prevAct `mul` w
                      desc del = tr prevAct `mul` del

-- | Predict the output for the given input, using the network represented by the given weights.
nnPredict :: Matrix Double  -- ^ The input as a row vector.
        -> [Matrix Double]  -- ^ The weights of the network.
        -> Matrix Double    -- ^ Returns an output prediction.

nnPredict inputActivation [] = inputActivation
nnPredict inputActivation (w:ws) = nnPredict (cmap sigma $ inputActivation `mul` w) ws
