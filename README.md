# USA-C2C-DistanceCalc

You will need the most recent sf file from [here](https://www.nber.org/research/data/county-distance-database)

You will also need the zip-county file from [here](https://www.huduser.gov/apps/public/uspscrosswalk/home)

The SF file will have distances from every county to every county, referenced by their FIPS codes (poor scientist who did this). We then want to make a matrix for mapping these counties codes to their city/state, get the distance, and repopulate the excel sheet.
