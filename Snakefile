#Run sequence_ml project

configfile: "config.yaml"
workdir: config["WORK_DIR"]

# Is executed until the provided file is created
rule all:
	input:
		#config["files"]["analysis"]["cad4_booltable_nocentr_resize"]
		#config["files"]["analysis"]["Figures"]["CAD4_overlapRatio_other.pdf"]
		#config["files"]["analysis"]["cad4_booltable_tisspec.gtf"]
		#config["files"]["analysis"]["Figures"]["DHS_overlapsCADs.pdf"]
		config["files"]["analysis"]["Figures"]["DHS_overlapsCADs_centr.pdf"]

#Test if snakemake works
rule testSnake:
	input:
		rscript = expand("src/R/vlad/{script}", script = config["Rscripts"]["TestRscript"]),
		f_in = config["testinput"]
	output:
		f_out = config["testout"]
	shell:
		"./{input.rscript} {input.f_in} {output.f_out}"