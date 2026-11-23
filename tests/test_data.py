"""Synthetic FMA fixture tests for Person A's metadata pipeline."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from album_similarity.data import (load_metadata, validate_audio_files, clean_genres,
    build_album_catalog, prepare_metadata, build_genre_representation,
    validate_catalog, run_pipeline)
from album_similarity.utils import read_table


def fixture_fma(root):
    metadata = root / 'data/raw/fma_metadata'
    audio = root / 'data/raw/fma_medium'
    metadata.mkdir(parents=True)
    audio.mkdir(parents=True)
    with (metadata / 'tracks.csv').open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['', 'album', 'album', 'artist', 'track', 'set'])
        writer.writerow(['track_id', 'id', 'title', 'id', 'genres', 'subset'])
        writer.writerows([
            [2, 10, 'Album One', 7, '[2]', 'small'],
            [3, 10, 'Album One', 7, '[3]', 'medium'],
            [4, 10, 'Album One', 7, '[]', 'medium'],
            [5, 20, 'Album Two', 8, '[99, 3]', 'medium'],
            [6, 20, 'Album Two', 8, '[3]', 'medium'],
        ])
    with (metadata / 'genres.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['genre_id','title','parent','top_level'])
        writer.writerows([[1,'Rock',0,1],[2,'Alternative',1,1],[3,'Electronic',0,3]])
    for tid in (2,3,4,5):
        name=f'{tid:06d}'
        target=audio / name[:3] / f'{name}.mp3'
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(b'ID3' + bytes(32))
    return metadata,audio


def config(min_tracks=3,parents=False):
    return {'schema_version':'1.0','module':'data','project_root':'..',
            'paths':{'audio_dir':'data/raw/fma_medium',
                     'metadata_dir':'data/raw/fma_metadata',
                     'output_dir':'data/prepared'},
            'min_tracks':min_tracks,'genres':{'include_parents':parents}}


def test_load_multilevel_fma_metadata(tmp_path):
    metadata,_=fixture_fma(tmp_path)
    raw=load_metadata(metadata)
    assert len(raw['tracks'])==5
    assert raw['tracks'][0]['genre_ids']==['2']
    assert raw['tracks'][0]['album_id']=='10'
    assert [g['column_index'] for g in raw['genres']]==[0,1,2]



def test_fma_medium_excludes_large_subset(tmp_path):
    metadata, audio = fixture_fma(tmp_path)

    with (metadata / "tracks.csv").open("a", newline="") as f:
        csv.writer(f).writerow(
            [7, 30, "Large Album", 9, "[2]", "large"]
        )

    raw = load_metadata(metadata)

    assert len(raw["tracks"]) == 5
    assert raw["report"]["metadata_excluded_subsets"]["large"] == 1
    assert "7" not in {t["track_id"] for t in raw["tracks"]}

    run_pipeline(config(), tmp_path)

    prepared = tmp_path / "data/prepared"
    report = json.loads(
        (prepared / "quality_report.json").read_text()
    )

    assert report["audio_exclusions"]["missing_audio"] == 1
    assert report["metadata_excluded_subsets"]["large"] == 1


def test_missing_audio_and_album_filter(tmp_path):
    metadata,audio=fixture_fma(tmp_path)
    raw=load_metadata(metadata)
    valid,exclusions=validate_audio_files(raw['tracks'], audio)
    assert len(valid)==4
    assert exclusions==[{'track_id':'6','reason':'missing_audio'}]
    cleaned,_=clean_genres(valid,raw['genres'])
    assert cleaned[-1]['genre_ids']==['3']
    albums,discarded=build_album_catalog(cleaned,3)
    assert [a['album_id'] for a in albums]==['10']
    assert discarded[0]['album_id']=='20'


def test_parent_genres_are_optional(tmp_path):
    metadata,_=fixture_fma(tmp_path)
    raw=load_metadata(metadata)
    off,_=clean_genres(raw['tracks'],raw['genres'],False)
    on,_=clean_genres(raw['tracks'],raw['genres'],True)
    assert off[0]['genre_ids']==['2']
    assert on[0]['genre_ids']==['1','2']


def test_end_to_end_output_matches_shared_contract(tmp_path):
    fixture_fma(tmp_path)
    manifest=run_pipeline(config(),tmp_path)
    prepared=tmp_path/'data/prepared'
    tracks=read_table(prepared/'tracks.csv','tracks')
    albums=read_table(prepared/'albums.csv','albums')
    genres=read_table(prepared/'genres.csv','genres')
    index=read_table(prepared/'album_genre_index.csv','album_genre_index')
    matrix=np.load(prepared/'album_genres.npy',allow_pickle=False)
    assert manifest['genre_matrix_shape']==[1,3]
    assert matrix.dtype==np.float32
    assert matrix.tolist()==[[0.0,1.0,1.0]]
    assert index==[{'row_index':'0','album_id':'10','has_genres':'true'}]
    assert albums[0]['track_ids']==['2','3','4']
    assert tracks[0]['audio_path']=='data/raw/fma_medium/000/000002.mp3'
    assert tracks[0]['genre_ids']==['2']
    assert len(genres)==3
    report=json.loads((prepared/'quality_report.json').read_text())
    assert report['audio_exclusions']['missing_audio']==1
    assert report['albums_below_min_tracks']==1
    assert report['validation']['valid'] is True
    album_metadata = json.loads(
        (prepared / "album_metadata.json").read_text()
    )
    assert album_metadata["10"]["track_count"] == 3
    assert album_metadata["10"]["track_ids"] == ["2", "3", "4"]


def test_deterministic_artifacts(tmp_path):
    fixture_fma(tmp_path)
    first=run_pipeline(config(),tmp_path)
    prepared=tmp_path/'data/prepared'
    old={x.name:x.read_bytes() for x in prepared.iterdir() if x.is_file()}
    second=run_pipeline(config(),tmp_path)
    assert first==second
    assert old=={x.name:x.read_bytes() for x in prepared.iterdir() if x.is_file()}


def test_min_tracks_must_be_explicit(tmp_path):
    fixture_fma(tmp_path)
    with pytest.raises(ValueError,match='min_tracks'):
        run_pipeline(config(min_tracks=None),tmp_path)


def test_invalid_mp3_signature_is_excluded(tmp_path):
    metadata,audio=fixture_fma(tmp_path)
    path=audio/'000/000004.mp3'
    path.write_bytes(b'not an mp3 file')
    valid,excluded=validate_audio_files(load_metadata(metadata)['tracks'],audio)
    assert {x['track_id'] for x in valid}=={'2','3','5'}
    assert {'track_id':'4','reason':'invalid_mp3_header'} in excluded


def test_catalog_rejects_mismatched_tracks(tmp_path):
    metadata,audio=fixture_fma(tmp_path)
    raw=load_metadata(metadata)
    valid,_=validate_audio_files(raw['tracks'],audio)
    cleaned,_=clean_genres(valid,raw['genres'])
    albums,_=build_album_catalog(cleaned,3)
    albums[0]['genre_ids']=['99']
    with pytest.raises(ValueError,match='genre IDs disagree'):
        validate_catalog([t for t in cleaned if t['album_id']=='10'],albums)


def test_report_counts_unknown_genre_assignments(tmp_path):
    fixture_fma(tmp_path)
    run_pipeline(config(min_tracks=1),tmp_path)
    report=json.loads((tmp_path/'data/prepared/quality_report.json').read_text())
    assert report['unknown_genre_assignments_removed']==1
    assert report['unknown_genre_ids']==['99']


def test_portable_output_dir(tmp_path):
    fixture_fma(tmp_path)
    settings=config()
    settings['paths']['output_dir']='some/other/location'
    run_pipeline(settings,tmp_path)
    rows=read_table(tmp_path/'some/other/location/tracks.csv','tracks')
    assert rows[0]['audio_path']=='data/raw/fma_medium/000/000002.mp3'


def test_cli_entrypoint_runs_with_yaml(tmp_path):
    import yaml
    from album_similarity.data import main
    fixture_fma(tmp_path)
    config_dir=tmp_path/'configs'
    config_dir.mkdir()
    (config_dir/'data.yaml').write_text(yaml.safe_dump(config(),sort_keys=False))
    result=main(['--config',str(config_dir/'data.yaml')])
    assert result['num_tracks']==3
    assert (tmp_path/'data/prepared/manifest.json').exists()
