mport argparse

parser = argparse.ArgumentParser(description='few-shot script %s')
parser.add_argument('--name', default='train_wiki',help='train_wiki/val_pubmed')
# parser.add_argument('--outpath', default='fwt',help='wiki/nyt/semeval/pubmed')

params=parser.parse_args()
print(params.name)
