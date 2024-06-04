import pandas as pd
import json
from collections import defaultdict
import os
from pathlib import Path
import pickle
import csv



data_folder_name = "data"


def _load_pickle(folder, filename):
    data = list()

    with open(os.path.join(os.path.dirname(__file__), '..' , data_folder_name, folder, filename), "rb") as f:
        done = False
        while not done:
            try:
                obj = pickle.load(f)
                data.append(obj)
            except EOFError:
                done = True
    return data


def _load_json(folder, filename):
    with open(os.path.join(os.path.dirname(__file__), "..", data_folder_name, folder, filename)) as f:
        return json.load(f)


def get_fig1a_json():
    return _load_json("fig1", "fig1a.json")


def get_fig1b_data():
    data = _load_json("fig1", "fig1b.json")
    table = list()
    for scheduler in ["tiresias", "themis", "yarn_fifo"]:
        table.append(list())
        for metric in ["mean_jct", "peak_unfairness", "mean_pred_error"]:
            table[-1].append(data[scheduler][metric])
    return table


def get_table1_data():
    data = _load_json("table1", "table1.json")
    table = list()
    for metric in ["mean_jct", "peak_unfairness", "mean_pred_error", "p99_pred_error"]:
        table.append(list())
        for scheduler in ["tiresias", "themis", "yarn_fifo"]:
            table[-1].append(data[scheduler][metric])
    return table

def get_fig4_data():
    data = _load_pickle("fig4", "figure4.pkl")
    assert (data[0]["X"] == data[1]["X"]).all() and (data[0]["Y"] == data[1]["Y"]).all()
    return data[1:]


def get_fig5a_data():
    data = _load_pickle("fig5", "figure5a.pkl")
    return {
        "afs": (data[0]["Y"] / 60).tolist(),
        "tiresias": (data[1]["Y"] / 60).tolist(),
        "flex_jct": (data[2]["Y"] / 60).tolist(),
    }


def get_fig5b_data():
    data = _load_pickle("fig5", "figure5b.pkl")
    return {
        "afs": data[0]["Y"].tolist(),
        "tiresias": data[1]["Y"].tolist(),
        "flex_jct": data[2]["Y"].tolist(),
    }


def get_fig6a_data():
    return _load_pickle("fig6", "figure6a.pkl")[:-1]


def get_fig6b_data():
    data = _load_pickle("fig6", "figure6b.pkl")
    order = [e["policy"] for e in data]
    output = dict()
    for i, e in enumerate(order):
        output[e] = {"mean_err_pred": data[i]["X"], "normalized_mean_jct": data[i]["Y"]}
    return output


def get_trace_data(table_file):
    with open(
        os.path.join(os.path.dirname(__file__), "..", data_folder_name, "fig7", table_file), "r", encoding="utf-8"
    ) as f:
        reader = csv.reader(f, delimiter=",")
        traces = list()
        data = defaultdict(list)
        for i, row in enumerate(reader):
            if i == 0:
                policies = list(map(lambda e: e, row[1:]))
            else:

                if skip_criteria(row[0]):
                    continue

                traces.append(row[0])
                for p, c in zip(policies, list(map(lambda e: round(float(e),1), row[1:]))):
                    data[p].append(c)
        new_data = dict()
        for k, v in data.items():
            if k != "FIFO":
                new_data[k] = v
        new_data["FIFO"] = data["FIFO"]

        print({"traces": traces, "data": new_data})

        return {"traces": traces, "data": new_data}




def get_fig8a_data():
    data = _load_pickle("fig8", "figure8a.pkl")
    return {
        "afs": data[0]["Y"].tolist(),
        "tiresias": data[1]["Y"].tolist(),
        "themis": data[2]["Y"].tolist(),
        "flex_jct": data[3]["Y"].tolist(),
    }


def get_fig8b_data():
    data = _load_pickle("fig8", "figure8b.pkl")
    order = [e["policy"] for e in data]
    output = dict()
    for i, e in enumerate(order):
        output[e] = {"p99_err_pred": data[i]["X"], "normalized_mean_jct": data[i]["Y"]}
    return output


def get_fig9a_data():
    return _load_json("fig9", "figure9a.json")

def get_fig9b_data():
    return _load_json("fig9", "figure9b.json")

def get_fig9c_data():
    return _load_json("fig9", "figure9c.json")

def get_fig10_data():
    data = dict()
    for scheduler in ["AFS", "FIFO", "PCS"]:
        fp = os.path.join(os.path.dirname(__file__), "..", "data", "fig10", f"error_analysis_{scheduler}")
        data[scheduler] = _read_error_analysis(fp)

    return data

def _read_error_analysis(fp):
    data = Path(fp).read_text()
    data = data.replace(".................", "")
    data = [e.strip() for e in data.split("fname:") if len(e.strip()) > 0]
    output = dict()
    for e in data:
        ls = e.split("\n")
        perc = int(ls[0][ls[0].index("_") + 1 : ls[0].index("p")])
        output[perc] = dict()
        for i in range(1, len(ls)):
            kv = ls[i].split(": ")
            output[perc][kv[0]] = float(kv[1])
    factor = output[0]["avg JCT"]
    for k in output:
        output[k]["avg JCT"] /= factor

    return output






def skip_criteria(s):

    check = ["#" in s,
        # "ee9e8c" in s,
        "2869ce" in s]

    return any(check)
