module Definitions where

-- | The logistic function as activation function.
sigma :: Double -> Double
sigma x = 1 / (1 + exp (-x))

-- | The derivative of the logistic function.
sigma' :: Double -> Double
sigma' x = exp (-x) / (1 + exp (-x))^2

-- | The mean sqaured error as loss function.
loss :: Double -> Double -> Double
loss actual expected = 1/2 * (actual - expected)^2
