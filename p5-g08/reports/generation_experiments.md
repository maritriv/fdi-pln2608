# Experimentos de generacion

- Checkpoint: `checkpoints/p5_causal_2608.pth`
- Prompt: `Alice was`
- Max new tokens: `40`

## Configuracion del checkpoint

| Parametro | Valor |
| --- | --- |
| `batch_size` | `64` |
| `context_size` | `128` |
| `d_model` | `128` |
| `dropout` | `0.1` |
| `epochs` | `5` |
| `learning_rate` | `0.0003` |
| `lr` | `0.0003` |
| `n_heads` | `4` |
| `n_layers` | `4` |
| `seq_len` | `128` |
| `task` | `causal_lm` |
| `vocab_size` | `300` |

## Generaciones

| Temperature | Top-k | Texto generado | Observaciones |
| ---: | ---: | --- | --- |
| 0.5 | 10 | Alice was a long way to find that they must have to do<br>that!”<br><br>“i haven’t tell you see, oh, if  |  |
| 0.5 | 20 | Alice was so long out of it, and all coming up in a<br>minute or two, with one else to see it as  |  |
| 0.5 | 50 | Alice was the words all good out again, coming it again.<br>she set the considering so much to be |  |
| 0.8 | 10 | Alice was a pawn out of it, and<br>she could not thought to herself, “if it would have a good must grine,  |  |
| 0.8 | 20 | Alice was so large bread-and-butter, who were snow, if it doesn’t<br>because the party all  |  |
| 0.8 | 50 | Alice was a comfort the pair of viege. she was very pant all in a large slow, but it<br>as the  |  |
| 1.2 | 10 | Alice was a cat; if you could, she was a bate, something one—” “surprised at that’s no<br>g |  |
| 1.2 | 20 | Alice was all thank croquet, which the time with so.<br><br>*      *      *      *      *      *<br><br>*      *      *      *      *<br><br>    *      *      *      *      *      *      *      *       |  |
| 1.2 | 50 | Alice was the meases?”<br><br>“by the march hare,”ite very poor much.<br><br>“only want _some_ nor _s |  |
