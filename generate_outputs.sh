#!/usr/bin/bash


for i in {1..50..5}; do   
	./accuracy 128 0.1 0 0 5 100 $i inputs/128.fluid outputs/128_step_$i.fluid
done

