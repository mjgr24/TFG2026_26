# Resumen de grid search: learning_rate + local_epochs

Generado: 2026-04-17T03:00:05

## Criterio recomendado

Elegir principalmente por **final_val_acc**, usando **final_val_loss** como desempate, revisando la estabilidad, y si dos opciones son muy parecidas, preferir la de menor **local_epochs**.

## Recomendación automática

Mejor candidato: **lr = 0.15**, **local_epochs = 5**  
- final_val_acc: 0.7134  
- final_val_loss: 0.8006  
- estabilidad: algo inestable

## Tabla resumen

| lr | local_epochs | status | final_val_acc | final_val_loss | best_val_acc | best_round | final_test_acc | estabilidad |
|---:|---:|---|---:|---:|---:|---:|---:|---|
| 0.001 | 5 | ok | 0.667 | 0.9167 | 0.667 | 5 | 0.6777 | muy estable |
| 0.005 | 5 | ok | 0.6688 | 0.761 | 0.6688 | 5 | 0.6722 | muy estable |
| 0.01 | 5 | ok | 0.6219 | 0.91 | 0.6219 | 5 | 0.6384 | muy estable |
| 0.02 | 5 | ok | 0.6823 | 0.9363 | 0.6823 | 5 | 0.6857 | muy estable |
| 0.05 | 5 | ok | 0.6179 | 0.964 | 0.6179 | 5 | 0.6423 | muy estable |
| 0.1 | 5 | ok | 0.7024 | 0.7621 | 0.7024 | 5 | 0.7101 | muy estable |
| 0.15 | 5 | ok | 0.7134 | 0.8006 | 0.7134 | 5 | 0.7264 | algo inestable |
| 0.2 | 5 | ok | 0.6867 | 0.7775 | 0.6867 | 5 | 0.6966 | muy estable |
| 0.25 | 5 | ok | 0.6653 | 0.8212 | 0.6653 | 5 | 0.6782 | bastante estable |
| 0.5 | 5 | ok | 0.5614 | 1.0164 | 0.5614 | 5 | 0.5787 | muy estable |

## Qué mirar mañana

1. Quédate primero con el mayor **final_val_acc**.
2. Si dos salen parecidos, elige el menor **final_val_loss**.
3. Si siguen muy parecidos, prefiere el más estable.
4. Si aún así están muy cerca, prefiere el menor **local_epochs**.
5. Usa **final_test_acc** solo como referencia final, no para tunear.
