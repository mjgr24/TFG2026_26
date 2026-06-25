# Resumen de grid search: learning_rate + local_epochs

Generado: 2026-04-16T21:42:28

## Criterio recomendado

Elegir principalmente por **final_val_acc**, usando **final_val_loss** como desempate, revisando la estabilidad, y si dos opciones son muy parecidas, preferir la de menor **local_epochs**.

## Recomendación automática

Mejor candidato: **lr = 0.05**, **local_epochs = 5**  
- final_val_acc: 0.502  
- final_val_loss: 1.3148  
- estabilidad: inestable

## Tabla resumen

| lr | local_epochs | status | final_val_acc | final_val_loss | best_val_acc | best_round | final_test_acc | estabilidad |
|---:|---:|---|---:|---:|---:|---:|---:|---|
| 0.02 | 5 | ok | 0.4583 | 1.2973 | 0.5232 | 4 | 0.4254 | inestable |
| 0.05 | 5 | ok | 0.502 | 1.3148 | 0.5411 | 1 | 0.4783 | inestable |

## Qué mirar mañana

1. Quédate primero con el mayor **final_val_acc**.
2. Si dos salen parecidos, elige el menor **final_val_loss**.
3. Si siguen muy parecidos, prefiere el más estable.
4. Si aún así están muy cerca, prefiere el menor **local_epochs**.
5. Usa **final_test_acc** solo como referencia final, no para tunear.
