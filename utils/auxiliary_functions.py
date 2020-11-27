import json


def set_json_attr_val(attr, data, output_dir):
    '''
    Sets attribute `attr` in the 'args.json' file to `data`
    '''
    with open(output_dir + 'args.json', "r") as jsonFile:
        json_data = json.load(jsonFile)


    json_data[attr] = data

    with open(output_dir + 'args.json', "w") as jsonFile:
        json.dump(json_data, jsonFile, sort_keys=True, indent=4)