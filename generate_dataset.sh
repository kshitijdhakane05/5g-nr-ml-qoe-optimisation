#!/bin/bash

scenario=1

for ue in 2 4 6 8 10
do
    for lambda in 5000 10000 20000 40000
    do
        for bw in 20 50
        do
            for size in 3 30 100
            do
                echo "Running scenario $scenario: UEs=$ue Lambda=$lambda BW=${bw}MHz Size=${size}m"

                ./ns3 run "qoe-dataset \
--ueNumPergNb=$ue \
--lambdaBe=$lambda \
--bandwidthBand1=${bw}e6 \
--scenarioSize=$size \
--scenarioId=$scenario"

                scenario=$((scenario + 1))
            done
        done
    done
done

echo "================================"
echo "All 120 scenarios completed."
echo "Dataset: qoe_dataset.csv"
echo "================================"
