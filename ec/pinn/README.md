# Program Induction Neural Networks
For training RobustFill-like networks (https://arxiv.org/pdf/1703.07469.pdf)

- Supports both output->program mode and input->output->program mode

Example: 3-shot learning a program `p:A->B` from a support set `X = [(a1,b1), (a2, b2), (a3, b3)]`
where `ai, bi, p` are sequences with vocabularies of `v_a, v_b, v_p`

```
from pinn import RobustFill
net = RobustFill(input_vocabularies=[v_a, v_b], target_vocabulary=v_f)
batch_inputs = [[(a1,b1), (a2, b2), (a3, b3)], ] // Batch * Support set * Num inputs/outputs/etc
batch_target = [p1, p2, p3, ...] // Batch size

score = net.optimiser_step(batch_inputs, batch_targets) //Single gradient update
samp = net.sample(batch_inputs) //Prediction
```

# Todo:
- [ ] Double check if correct: attend during P->FC rather than during softmax->P?
- [X] Output attending to input
- [X] Target attending to output
- [X] Allow both input->target and (input,output)->target modes
- [ ] Pytorch JIT
- [ ] BiLSTM
- [ ] Multiple attention and different attention functions
- [ ] Reinforce
- [ ] Beam search
- [ ] Give n_examples as input to FC
