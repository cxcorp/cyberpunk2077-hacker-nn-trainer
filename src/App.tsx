import React, {
  useCallback,
  useState,
  useRef,
  useEffect,
  useMemo,
} from "react";

import maxBy from "lodash/maxBy";

import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-wasm";
import TF from "@tensorflow/tfjs";
import * as KNNClassifier from "@tensorflow-models/knn-classifier";
import * as MobileNet from "@tensorflow-models/mobilenet";

import "react-dropzone-uploader/dist/styles.css";
import Dropzone from "react-dropzone-uploader";

import tilesJson from "./tiles.json";
import tests from "./tests.json";
import pretrains from "./pretrains";
import "./App.css";

import pixelsJson from "./pixels.json";
const pixels = pixelsJson as { tile: string; pixels: number[][] }[];

interface LabelableImageProps {
  i: number;
  src: string;
  onClick: (i: number) => void;
  background?: string;
}

const LabelableImage = ({
  onClick,
  i,
  src,
  background,
}: LabelableImageProps) => {
  const handleclick = useCallback(() => {
    onClick(i);
  }, [i, onClick]);
  return (
    <div
      style={{ background: background || "white" }}
      className="labelable-image"
      onClick={handleclick}
    >
      <img src={src} />
    </div>
  );
};

const tiles = tilesJson as string[];
const testTiles = [
  "Cyberpunk-hacking-minigame-3-1212x682-0.bmp",
  "Cyberpunk-hacking-minigame-3-1212x682-1.bmp",
  "Cyberpunk-hacking-minigame-3-1212x682-10.bmp",
  "Cyberpunk-hacking-minigame-3-1212x682-11.bmp",
  "Cyberpunk-hacking-minigame-3-1212x682-12.bmp",
  "Cyberpunk-hacking-minigame-3-1212x682-13.bmp",
  "Cyberpunk-hacking-minigame-3-1212x682-14.bmp",
  "Cyberpunk-hacking-minigame-3-1212x682-15.bmp",
  "Cyberpunk-hacking-minigame-3-1212x682-16.bmp",
  "Cyberpunk-hacking-minigame-3-1212x682-17.bmp",
  "Cyberpunk-hacking-minigame-3-1212x682-18.bmp",
  "Cyberpunk-hacking-minigame-3-1212x682-19.bmp",
  "Cyberpunk-hacking-minigame-3-1212x682-2.bmp",
  "Cyberpunk-hacking-minigame-3-1212x682-20.bmp",
  "Cyberpunk-hacking-minigame-3-1212x682-21.bmp",
  "Cyberpunk-hacking-minigame-3-1212x682-22.bmp",
  "Cyberpunk-hacking-minigame-3-1212x682-23.bmp",
  "Cyberpunk-hacking-minigame-3-1212x682-24.bmp",
  "Cyberpunk-hacking-minigame-3-1212x682-3.bmp",
  "Cyberpunk-hacking-minigame-3-1212x682-4.bmp",
  "Cyberpunk-hacking-minigame-3-1212x682-5.bmp",
  "Cyberpunk-hacking-minigame-3-1212x682-6.bmp",
  "Cyberpunk-hacking-minigame-3-1212x682-7.bmp",
  "Cyberpunk-hacking-minigame-3-1212x682-8.bmp",
  "Cyberpunk-hacking-minigame-3-1212x682-9.bmp",
].map((f) => `/testfiles/${f}`); //tests as string[];

type Opt = "55" | "7A" | "BD" | "1C" | "E9";
const opts: Opt[] = ["55", "7A", "BD", "1C", "E9"];
const colors = ["#3498db", "#9b59b6", "#e74c3c", "#e67e22", "#f1c40f"];
const optToColor: Record<Opt, string> = opts.reduce((acc, val, i) => {
  acc[val] = colors[i];
  return acc;
}, {} as Record<Opt, string>);

const OptBtn = ({
  opt,
  i,
  onClick,
  children,
  current,
}: {
  opt: string;
  i: number;
  onClick: (i: number) => void;
  current: boolean;
  children: React.ReactChild;
}) => {
  const hc = useCallback(() => {
    onClick(i);
  }, [onClick, i]);
  return (
    <button
      style={{ background: current ? "white" : colors[i] }}
      key={opt}
      onClick={hc}
      className="control-btn"
    >
      {children}
    </button>
  );
};

const getLabelIdxToOptIdxInitialState = (): {
  [i: number]: number | undefined;
} => {
  return pretrains.reduce((acc, curr) => {
    const { img, opt } = curr;
    return { ...acc, [tiles.indexOf(img)]: opts.indexOf(opt as Opt) };
  }, {});
};

const imgToColumnSum = (imgSrc: string): Promise<number[]> => {
  return new Promise((resolve, reject) => {
    const img = document.createElement("img");
    img.src = imgSrc;

    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.width;
      canvas.height = img.height;

      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0);
      document.body.removeChild(img);

      const out: number[] = [];

      const idata = ctx.getImageData(0, 0, img.width, img.height);
      for (let y = 0; y < img.height; y++) {
        let colSum = 0;
        for (let x = 0; x < img.width; x += 4) {
          const r = idata.data[y * (img.width * 4) + x * 4];
          colSum += 255 - r;
        }
        out.push(colSum);
      }

      resolve(out);
    };
    img.onerror = (e) => {
      reject(e);
    };
    document.body.appendChild(img);
  });
};

/*
const imgToPixels = (imgSrc: string): Promise<number[][]> => {
  return new Promise((resolve, reject) => {
    const img = document.createElement("img");
    img.src = imgSrc;

    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.width;
      canvas.height = img.height;

      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0);
      document.body.removeChild(img);

      const out: number[][] = new Array(img.height);

      const idata = ctx.getImageData(0, 0, img.width, img.height);
      for (let y = 0; y < img.height; y++) {
        const row = new Array(img.width);
        for (let x = 0; x < img.width; x++) {
          row[x] =
            255 -
            Math.floor(
              0.2126 * idata.data[y * (img.width * 4) + x * 4] +
                0.7152 * idata.data[y * (img.width * 4) + (x + 1) * 4] +
                0.0722 * idata.data[y * (img.width * 4) + (x + 2) * 4]
            ) -
            1;
          if (isNaN(row[x])) {
            row[x] = 0;
          }
        }
        out[y] = row;
      }

      resolve(out);
    };
    img.onerror = (e) => {
      reject(e);
    };
    document.body.appendChild(img);
  });
};

const dumpImgPixels = async () => {
  const out: { tile: string; pixels: number[][] }[] = [];

  let dump = true;
  await Promise.all(
    tiles.map(async (tile) => {
      const pixels = await imgToPixels(tile);
      out.push({ tile, pixels });
    })
  );

  console.log(JSON.stringify(out));
};

const calcColSumsForTiles = async (): Promise<{ [img: string]: number[] }> => {
  const lookup: { [img: string]: number[] } = {};
  let i = 0;
  await Promise.all(
    tiles.map(async (tile) => {
      const coldata = await imgToColumnSum(tile);
      console.log(`${i++}/${tiles.length}`);
      lookup[tile] = coldata;
    })
  );
  return lookup;
};*/

interface VerifyImageProps {
  label: string;
  imgSrc: string;
  color: string;
}
const VerifyImage = ({ label, imgSrc, color }: VerifyImageProps) => {
  return (
    <div style={{ background: color || "white" }} className="verify-image">
      <span className="verify-image__label">{label}</span>
      <img src={imgSrc} />
    </div>
  );
};

async function withImg<T>(
  imgSrc: string,
  cb: (img: HTMLImageElement) => T
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const img = document.createElement("img");
    img.src = imgSrc;

    img.onload = () => {
      const res = cb(img);
      document.body.removeChild(img);
      resolve(res);
    };
    img.onerror = (e) => {
      reject(e);
    };
    document.body.appendChild(img);
  });
}

function shuffle<T>(array: T[]): T[] {
  var currentIndex = array.length,
    temporaryValue,
    randomIndex;

  // While there remain elements to shuffle...
  while (0 !== currentIndex) {
    // Pick a remaining element...
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex -= 1;

    // And swap it with the current element.
    temporaryValue = array[currentIndex];
    array[currentIndex] = array[randomIndex];
    array[randomIndex] = temporaryValue;
  }

  return array;
}

async function withImgs<T>(
  imgSrcs: string[],
  cb: (imgs: HTMLImageElement[]) => Promise<T>
): Promise<T> {
  return new Promise<T>(async (resolve, reject) => {
    const imgs = imgSrcs.map((src) => {
      const img = document.createElement("img");
      img.src = src;
      return img;
    });

    const loadPromises = imgs.map(
      (img) =>
        new Promise<HTMLImageElement>((resolve, reject) => {
          img.onload = () => {
            resolve(img);
          };
          img.onerror = (e) => reject(e);
        })
    );

    imgs.forEach((img) => document.body.appendChild(img));

    const doms = await Promise.all(loadPromises);
    const ret = await cb(doms);

    imgs.forEach((img) => document.body.removeChild(img));
    resolve(ret);
  });
}

let mobilenet: MobileNet.MobileNet | undefined;
let classifier: KNNClassifier.KNNClassifier | undefined;

const Tiler = () => {
  /*useEffect(() => {
    async function doo() {

      await withImgs(verifySet, async (imgs) => {
        console.log("loaded, verifying...");
        for (let i = 0; i < imgs.length; i++) {
          console.log(i);
          const xlogits = mobilenet!.infer(imgs[i]);
          const result = await classifier!.predictClass(xlogits);
          //console.log(`${result.label} - ${verifySet[i]}`);
          classifieds.push({ label: result.label, imgSrc: verifySet[i] });
        }
      });

    }
    doo();
  }, []);*/

  interface RResult {
    columns: number;
    rows: number;
    labels: string[][];
  }
  const [result, setResult] = useState<RResult>();
  const [clearFiles, setClearfiles] = useState<() => {}>(() => () => {});

  return (
    <div>
      <Dropzone
        addClassNames={{ dropzone: "dropzone", preview: "dropzone-preview" }}
        onSubmit={async (files, allFiles) => {
          const imgs = await Promise.all(
            allFiles.map((file) => {
              return new Promise<{
                img: HTMLImageElement;
                coords: { x: number; y: number };
              }>((resolve, reject) => {
                let alreadyRejected = false;

                const fr = new FileReader();

                fr.onload = () => {
                  const img = document.createElement("img");
                  img.onload = () => {
                    const fname = file.meta.name.split(".")[0];
                    const [y, x] = fname.split("-").map((s) => parseInt(s, 10));
                    console.log(fname);

                    resolve({ img, coords: { x, y } });
                  };
                  img.onerror = () => {
                    if (!alreadyRejected) {
                      reject();
                      alreadyRejected = true;
                    }
                  };
                  img.src = fr.result as string;
                };

                fr.onerror = () => {
                  if (!alreadyRejected) {
                    reject();
                    alreadyRejected = true;
                  }
                };

                fr.readAsDataURL(file.file);
              });
            })
          );

          const rowCount = maxBy(imgs, (img) => img.coords.y)!.coords.y + 1;
          const columnCount = maxBy(imgs, (img) => img.coords.x)!.coords.x + 1;
          const out = new Array(columnCount);
          for (let i = 0; i < out.length; i++) {
            out[i] = new Array(rowCount);
          }

          for (const { coords, img } of imgs) {
            const xlogits = mobilenet!.infer(img);
            const result = await classifier!.predictClass(xlogits);
            console.log({
              label: result.label,
              confidences: result.confidences,
            });
            out[coords.y][coords.x] = result.label;
          }

          setResult({
            columns: columnCount,
            rows: rowCount,
            labels: out,
          });

          setClearfiles(() => () => {
            allFiles.forEach((f) => f.remove());
          });
        }}
        inputContent="drop tiles here"
        submitButtonDisabled={(files) => files.length < 1}
      />
      <div>
        <button onClick={clearFiles}>clear</button>
      </div>
      <div>
        <pre>
          <code>
            {result &&
              result.labels
                .map((row) => row.join("  "))
                .map((row, i) => (
                  <React.Fragment key={[row, i].join("")}>
                    {row}
                    <br />
                  </React.Fragment>
                ))}
          </code>
        </pre>
      </div>
    </div>
  );
};

function App() {
  const [tileColSum, setTileColSum] = useState<{ [img: string]: number[] }>();
  const [tileColSumRdy, setTileColSumRdy] = useState<boolean>(false);

  /*useEffect(() => {
    setTimeout(() => {
      dumpImgPixels()
        .then(() => console.log("dumped pixels"))
        .catch((e: any) => console.error(e));
    }, 200);
  }, []);*/

  /*useEffect(() => {
    setTimeout(() => {
      calcColSumsForTiles().then((res) => {
        setTileColSum(res);
        setTileColSumRdy(true);
      });
    }, 100);
  }, [setTileColSumRdy, setTileColSum]);*/

  const [classifiedImages, setClassifiedImages] = useState<
    {
      label: string;
      imgSrc: string;
    }[]
  >([]);

  const [mbnetReady, setmbnetReady] = useState<number>(0);

  useEffect(() => {
    async function doo() {
      //TF.registerBackend('webgl', () => TFBackendWebGL)
      const trainset = shuffle([...pretrains]);

      classifier = KNNClassifier.create();
      mobilenet = await MobileNet.load();

      console.log("loading images");
      await withImgs(
        trainset.map((t) => t.img),
        async (imgs) => {
          console.log("finish loading images, training...");
          for (let i = 0; i < imgs.length; i++) {
            console.log(`${i}/${imgs.length}`);
            const logits = mobilenet!.infer(imgs[i]);
            classifier!.addExample(logits, trainset[i].opt);
          }
        }
      );

      console.log(classifier!.getClassifierDataset());

      console.log("loading verification images");
      const verifySet = testTiles;

      let classifieds: {
        label: string;
        imgSrc: string;
      }[] = [];

      await withImgs(verifySet, async (imgs) => {
        console.log("loaded, verifying...");
        for (let i = 0; i < imgs.length; i++) {
          console.log(i);
          const xlogits = mobilenet!.infer(imgs[i]);
          const result = await classifier!.predictClass(xlogits);
          //console.log(`${result.label} - ${verifySet[i]}`);
          classifieds.push({ label: result.label, imgSrc: verifySet[i] });
        }
      });

      setClassifiedImages(classifieds);
      setmbnetReady((n) => n + 1);
    }
    doo();
  }, [setClassifiedImages]);

  const [currOpt, setCurrOpt] = useState<number>(0);
  const [labelIdxToOptIdx, setLabelIdxToOptIdx] = useState<{
    [i: number]: number | undefined;
  }>(getLabelIdxToOptIdxInitialState());

  const currOptRef = useRef<number>(currOpt);
  const handleOptClick = useCallback(
    (i: number) => {
      currOptRef.current = i;
      setCurrOpt(i);
    },
    [setCurrOpt]
  );

  const handleTileClick = useCallback(
    (i: number) => {
      setLabelIdxToOptIdx((curr) => ({ ...curr, [i]: currOptRef.current }));
    },
    [setLabelIdxToOptIdx, currOptRef]
  );

  const labelIdxToOptIdxRef = useRef(labelIdxToOptIdx);
  labelIdxToOptIdxRef.current = labelIdxToOptIdx;
  const doDump = useCallback(() => {
    console.log(
      JSON.stringify(
        Object.entries(labelIdxToOptIdxRef.current).map(
          ([labelIdx, optIdx]) => ({
            img: tiles[parseInt(labelIdx, 10)],
            opt: opts[optIdx!],
          })
        )
      )
    );
  }, [labelIdxToOptIdxRef]);

  const renderOrder = useMemo<number[]>(() => {
    if (!tileColSumRdy) {
      return [];
    }

    const tileIdxToRenderIdx: number[] = [];
    tileIdxToRenderIdx.length = tiles.length;
    [...tiles]
      .map((tile, i) => ({ tile, i }))
      .sort((a, b) => {
        const colA = tileColSum![a.tile];
        const colB = tileColSum![b.tile];

        let cum = 0;
        for (let y = 0; y < colA.length; y++) {
          cum += colA[y] - colB[y];
        }
        return cum;
      })
      .forEach((v, renderI) => {
        tileIdxToRenderIdx[v.i] = renderI;
      });
    return tileIdxToRenderIdx;
  }, [tileColSumRdy]);

  return (
    <div className="App">
      <div className="controls">
        {opts.map((opt, i) => (
          <OptBtn
            key={opt}
            current={currOpt === i}
            i={i}
            opt={opt}
            onClick={handleOptClick}
          >
            {opt}
          </OptBtn>
        ))}
      </div>
      <div>
        <button onClick={doDump}>dump</button>
      </div>
      <div>
        <Tiler key={"tiler" + mbnetReady} />
      </div>
      {classifiedImages.length < 0 && <p>Classifying...</p>}
      <div className="tile-container">
        {classifiedImages.map(({ imgSrc, label }) => (
          <VerifyImage
            key={imgSrc}
            label={label}
            imgSrc={imgSrc}
            color={optToColor[label as Opt]}
          />
        ))}
      </div>
      <div className="tile-container">
        {tiles
          .map((tile, tileI) => {
            if (renderOrder.length > 0) {
              return { tile, tileI, renderI: renderOrder[tileI] };
            } else {
              return { tile, tileI, renderI: tileI };
            }
          })
          .sort((a, b) => a.renderI - b.renderI)
          .map(({ tile, tileI, renderI }) => (
            <LabelableImage
              key={tile}
              i={tileI}
              src={tile}
              background={
                typeof labelIdxToOptIdx[tileI] === "undefined"
                  ? undefined
                  : colors[labelIdxToOptIdx[tileI]!]
              }
              onClick={handleTileClick}
            />
          ))}
      </div>
    </div>
  );
}

export default App;
