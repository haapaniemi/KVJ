
Hyytiälä Autumn School 2024 
Analysis of atmosphere-surface interactions and feedbacks

-------------------------------------------------------------
	Kuivajärvi group's analysis scripts
-------------------------------------------------------------

Authors

        Marta Fregona (marta.fregona@helsinki.fi)
        Anni Karvonen (anni.karvonen@helsinki.fi)
        Pauline Ogola (pauline.ogola@helsinki.fi)
        Gunnar Thorsen Liahjell (gunnartl@uio.no)
        Eevi Silvennoinen (eevi.silvennoinen@helsinki.fi)
        Veera Haapaniemi (veera.haapaniemi@fmi.fi)


Input datasets

	Kohonen, K.-M., Mammarella, I., Ojala, A., Laakso, H., Matilainen, T., Salminen, T., 
	Levula, J., Ala-Könni, J., Kolari, P., et al.: SMEAR II Lake Kuivajärvi meteorology,
	water quality and eddy covariance, https://doi.org/10.23729/e085f3d1-b18a-46a1-aaa6-cf89bad1f647,
	University of Helsinki, Institute for Atmospheric and Earth System Research, 2024


	Gap filled CO2 datasets from I. Mammarella and J. Ala-Könni (unpublished).


Description


        code
        -------------------------------------------------------
        * turnoverPeriodStripes.py (Gunnar)
		plotting timing of all turnovers in a calendar year

	* length_and_timing.py (Anni)
		plotting turnover length and timing
        * meteorology_correlation.py (Anni)
                plotting turnover length correlation with meteorological variables

	* CO2_gapfilled_fluxes_IM/JAK.ipynb (Marta)
                contain plots of the fluxes and calculations for shoulder seasons contribution to the annual CO2 budget
                IM=boundary layer method for gap filling
                JAK=random forest
	* Lake_Analyzer_Master_Plots.ipynb (Marta)
                contains plots and calculation of yearly average thermocline depth
	* AvgYearTemp.ipynb (Marta)
                contains calculation of yearly average water temperature
        * Convert_waterT.ipynb (Marta)
                contains calculations and plots for the definition of turnover periods
        
        * plot_heat_contents_n_winds.py (Veera)
        * plot_gas_transfer_velocity.py (Veera)

        data
        -------------------------------------------------------
        * data_in
        * data_out


        image
        -------------------------------------------------------
        * saving generated images here
