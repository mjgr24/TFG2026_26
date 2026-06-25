# Resumen de grid search: learning_rate + local_epochs

Generado: 2026-04-16T20:24:32

## Criterio recomendado

Elegir principalmente por **final_val_acc**, usando **final_val_loss** como desempate, revisando la estabilidad, y si dos opciones son muy parecidas, preferir la de menor **local_epochs**.

## Recomendación automática

Mejor candidato: **lr = 0.01**, **local_epochs = 5**  
- final_val_acc: 0.4957  
- final_val_loss: 1.2917  
- estabilidad: bastante estable

## Tabla resumen

| lr | local_epochs | status | final_val_acc | final_val_loss | best_val_acc | best_round | final_test_acc | estabilidad |
|---:|---:|---|---:|---:|---:|---:|---:|---|
| 0.001 | 5 | ok | 0.4132 | 1.3653 | 0.4952 | 3 | 0.3866 | inestable |
| 0.005 | 5 | ok | 0.4382 | 1.2645 | 0.5084 | 1 | 0.3993 | inestable |
| 0.01 | 5 | ok | 0.4957 | 1.2917 | 0.5025 | 4 | 0.4656 | bastante estable |

## Qué mirar mañana

1. Quédate primero con el mayor **final_val_acc**.
2. Si dos salen parecidos, elige el menor **final_val_loss**.
3. Si siguen muy parecidos, prefiere el más estable.
4. Si aún así están muy cerca, prefiere el menor **local_epochs**.
5. Usa **final_test_acc** solo como referencia final, no para tunear.
