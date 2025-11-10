SCADA Data from Aventa AV-7 (6kW) Research Wind Turbine, located in Taggenberg (ZH), Lat:47.52 Long:8.68236, owned and operated by IET-OST. The data covers time period from 2025-01-01 to 2025-10-15, sampled at 1Hz, and includes operation with the imposed nacelle yaw offsets of 4 and 6 degrees.

The present dataset is prepared for the static yaw offset identification challenge (see Related works). The training dataset contains labeled data (imposed yaw offset) and 11 SCADA Channels containing time-series observations (at 1Hz) of: Rotor speed in RPM, Generator speed in RPM, Stator temperature in C, Wind speed in m/s, Converter active power in kW, Wind direction w.r.t. nacelle measured in degrees, System supply voltage 24V , Pitch angle in degrees, Turbine status. The test dataset contains 1-hour long segments taken out of the original data, shuffled around and does not have the labels. Additionally the original date-time labels have been replaced by relative time (t=0 at the start of each segment). Each segment is labeled with unique id.  The model performance on test data is used for create a leader board during the challenge.

Further information is available upon request.

This dataset for the Static Yaw Ground Truth Challenge can be accessed on Zenodo here.

The data is described with a metadata.croissant.json file. The file is in ML Croissant format. You can get all the information about this specification here.

The data itself is stored as two Apache Parquet files:  "Training" and "Test". The "Training" data set includes three known static yaw offsets 0°, 4° and 6°.  The test data contains 1 hour long segments, from the original data, shuffled around. This is used for the leaderboard. Hence the date-time is obscured, and replaced with relative time (t=0 at the start of  each segment). Moreover, the static yaw labels  are removed,  while row_id and segment_id columns are added in.

NOTE: While the train data is biased towards stretches of time without any wind, the test data has been selected to have a uniform distribution across different operating conditions (below cut-in, around rated, rated to cutout, etc..)

We have also removed some data for a private test, which will be performed at the end of the challenge for a final score, to avoid hyper-parameter over-fitting. The private test data  in addition to   0°, 4° and 6° offset data has data with a "mystery" offset, not seen in the public data, to test how well the model generalises.