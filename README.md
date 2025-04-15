📁 non_countries.csv

This file contains height data for regions or sub-national areas that are not considered sovereign countries. These might be urban/rural divisions, provinces, or cities. (source: Wikipedia

    Columns:

        country: Name of the region or area (e.g. Brazil – Urban)

        male_height_cm: Average male height in centimetres

        female_height_cm: Average female height in centimetres

    Examples:

        Brazil – Urban: Male 173.5 cm, Female 161.6 cm

        Costa Rica – San José: Male 169.4 cm, Female 155.9 cm

📁 all_countries_nation_height.csv

This is your master country dataset, combining ISO codes, regions, nationalities, and average heights.

    Columns:

        country: Standard country name

        alpha_2, alpha_3: ISO Alpha-2 and Alpha-3 codes

        male_avg, female_avg: Estimated average heights

        nationality: Associated demonym (e.g. Albanian)

        region, sub-region, intermediate-region: UN-style geographic classifications

    Examples:

        Albania: Male 174.1 cm, Female 162.2 cm, Southern Europe

        Algeria: Male 175.0 cm, Female 162.3 cm, Northern Africa

        Afghanistan: Male 168.5 cm, Female 156.1 cm, Southern Asia

Source: Wikipedia contributors. "Average human height by country." *Wikipedia, The Free Encyclopedia*. https://en.wikipedia.org/wiki/Average_human_height_by_country
