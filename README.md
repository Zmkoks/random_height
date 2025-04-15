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


📊 Overview of the Data Sources

WHO Child Growth Standards (0–5 years): These standards were developed from the WHO Multicentre Growth Reference Study, which collected growth data from over 8,000 healthy breastfed infants and young children across diverse countries, including Brazil, Ghana, India, Norway, Oman, and the USA. The resulting charts provide percentile curves for various growth indicators, such as length/height-for-age, weight-for-age, and BMI-for-age, and are considered the global standard for assessing child growth from birth to 5 years. ​
World Health Organization (WHO)+2World Health Organization (WHO)+2World Health Organization (WHO)+2

CDC Growth Charts (2–20 years): The CDC's 2000 growth charts offer percentile curves for U.S. children and adolescents aged 2 to 20 years, covering measurements like stature-for-age, weight-for-age, and BMI-for-age. These charts are widely used in the United States to monitor growth patterns and assess the nutritional status of children and teens. ​
CDC+1CDC+1
🗂 Structure of Your Simplified Growth Chart

Your consolidated chart includes the following columns:​

    sex

    age

    P5

    P10

    P25

    P50

    P75

    P90

    P95​
    CDC+5CDC+5CDC+5
    World Health Organization (WHO)

The rows represent specific age milestones:​

    Birth

    3 months

    6 months

    9 months

    1 year

    1.5 years

    2 years, continuing in half-year increments up to 18 years​
    World Health Organization (WHO)+1CDC+1
    World Health Organization (WHO)

This format provides a streamlined reference for tracking growth patterns across key developmental stages.​
🔗 References

    WHO Child Growth Standards: https://www.who.int/tools/child-growth-standards

    CDC Growth Charts: https://www.cdc.gov/growthcharts/index.htm​
    World Health Organization (WHO)+3World Health Organization (WHO)+3

