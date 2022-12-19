from lxml import etree, objectify
from lxml.etree import SubElement

def gen_txt(bbox,filename=None,save_path=None):
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('dgface'),
        E.filename(filename),
        E.source(
        E.database('ljw'),
        E.annotation('dgface'),
        E.image('dgface detection'),
        E.flickrid("0")
        ),
        E.size(
        E.width('1920'),
        E.height('1080'),
        E.depth('3')
        ),
        )
    for (x0,y0,x1,y1) in bbox:
        object_tree = E.object(
            E.name('face'),
            E.pose('Unspecified'),
            E.truncated('0'),
            E.bndbox(
                E.xmin('{}'.format(x0)),
                E.ymin('{}'.format(y0)),
                E.xmax('{}'.format(x1)),
                E.ymax('{}'.format(y1)))
            )
        anno_tree.append(object_tree)
    etree.ElementTree(anno_tree).write(save_path, pretty_print=True)


def main():
    bboxs = [[1,2,3,4],[4,5,6,7]]
    gen_txt(bboxs,'1.jpg','tmp/test.xml')

if __name__ == "__main__":
    main()