○OpenPoseとは

画像内に映る、複数人の姿勢を特定する

semantic segmatonはピクセルごとにクラス分類する.
pose esitimationではピクセルごとに回帰する.

方法としては一つ 
●物体検知してから姿勢推定  複数人数の処理には時間がかかる
●PAFs(部位のつながりを示す指標)を用いてリンクさせる
    例えば左肘と左手首をリンクさせるとき間の関節のクラスを一つ用意する.

○OpenPoseフロー
    1.画像のリサイズ(368x368)と色素標準化
    2.networkに画像入力,
      出力は[batch_size, 部位class数(19), height(368), width(368)]
            [batch_size, リンクclass数(38), height(368), width(368)]

      部位は部位18箇所+Noneの19class 
      PAFsはリンク37+Noneの38class
    3.部位をピクセルごとに決め,リンク繋げる.