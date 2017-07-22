import coremltools

def convertModel(model, title, description = None, class_labels = None):
    coreml_model = coremltools.converters.keras.convert(model,
                                                        input_names="image",
                                                        image_input_names="image",
                                                        output_names=["output", "classLabel"],
                                                        class_labels=class_labels)
    coreml_model.author = "Will McGinty"
    coreml_model.license = "BSD"
    coreml_model.short_description = description

    coreml_model.input_description["image"] = "A 28x28 grayscale image of an alphanumeric character."
    coreml_model.output_description["output"] = "The probability of each character."
    coreml_model.output_description["classLabel"] = "The most likely character."

    coreml_model.save(title + ".mlmodel")
