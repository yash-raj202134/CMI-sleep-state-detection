# CMI Sleep State Detection

This project focuses on the development of a machine learning model to detect sleep onset and wakefulness from wrist-worn accelerometer data. By accurately detecting periods of sleep and wakefulness, researchers can conduct more reliable, large-scale sleep studies, leading to a deeper understanding of sleep patterns and disturbances across various populations.

## Description

The project aims to enhance the accuracy of sleep monitoring by leveraging data science and machine learning techniques. This approach allows researchers to better analyze wrist-worn accelerometer data and improve sleep monitoring methods.

The data used in this project comes from the Healthy Brain Network, a landmark mental health study based in New York City that provides data to help children around the world. This project is supported by the Stavros Niarchos Foundation (SNF) through the SNF Global Center for Child and Adolescent Mental Health at the Child Mind Institute.

## Context

Sleep is essential for overall health and cognitive functioning. However, traditional methods of sleep research, such as sleep logs, can be challenging to use reliably. This project uses accelerometer data to detect sleep onset and wake events, providing researchers with more naturalistic data capture and accurate annotation.

## Dataset Description

The dataset consists of approximately 500 multi-day recordings of wrist-worn accelerometer data annotated with two event types: sleep onset and wakeup. The goal is to detect these events in the accelerometer data series. Key aspects of the dataset include:

- A single sleep period must be at least 30 minutes in length.
- Sleep periods can be interrupted by bouts of activity that do not exceed 30 consecutive minutes.
- No sleep windows can be detected unless the watch is deemed to be worn for the duration.
- The longest sleep window during the night is the only one recorded.
- If no valid sleep window is identifiable, neither an onset nor a wakeup event is recorded for that night.
- Sleep events do not need to straddle the day-line, allowing multiple sleep windows within a given period.

The data series represents a continuous (multi-day/event) recording for a unique experimental subject. Periods when the accelerometer device was removed are excluded from event predictions.

## How to Use

1. **Data Preparation**: Load the accelerometer data and preprocess it according to the dataset description.
2. **Model Development**: Train a machine learning model using the annotated data to detect sleep onset and wakeup events.
3. **Evaluation**: Evaluate the model's performance using appropriate metrics such as precision, recall, and F1-score.
4. **Implementation**: Use the trained model to predict sleep onset and wakeup events in new data series.

## Acknowledgments

The data used for this project was provided by the Healthy Brain Network, a landmark mental health study based in New York City that will help children around the world. This project is supported by the Stavros Niarchos Foundation (SNF) as part of its Global Health Initiative (GHI) through the SNF Global Center for Child and Adolescent Mental Health at the Child Mind Institute.

## License

This project is licensed under the MIT License.
