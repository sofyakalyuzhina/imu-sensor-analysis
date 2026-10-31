# Данные

Источник: [Industrial Robotic Arm — IMU Data (CASPER 1 & 2), Hakan Kayan](https://www.kaggle.com/datasets/hkayan/industrial-robotic-arm-imu-data-casper-1-and-2).


Для воспроизведения нужно положить предоставленный файл `IMU_10Hz.csv` в эту папку:

```text
data/
├── README.md
└── IMU_10Hz.csv    # локальный файл, не коммитить
```

В текущем анализе использована запись из 874 937 строк одного модуля IMU8.

Характеристики проверенной версии:

- Размер: 135 748 396 байт (около 129.46 MiB).
- SHA-256: `1221df9be80c229969f3f21f820837e68ad28f4ccb3949bf4013f00d6d69d62c`.
- Поля: `name,time,accX,accY,accZ,gyroX,gyroY,gyroZ,magX,magY,magZ`.
- Время хранится целыми числами в наносекундах; временные метки строго возрастают.
- Единицы сенсоров в ноутбуке — рабочие предположения исходного практикума, не подтверждённые метаданными CSV.



