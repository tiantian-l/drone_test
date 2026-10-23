"""Dependency-free evaluation accounting, testable without a simulator."""


class EpisodeLedger:
    def __init__(self, workers, quota, stride, seed_base):
        if min(workers, quota) < 1 or stride < workers:
            raise ValueError('Positive quotas and stride >= workers are required')
        self.workers, self.quota, self.stride, self.seed_base = workers, quota, stride, seed_base
        self.counts = [0] * workers
        self.lengths = [0] * workers
        self.records, self.maps = [], []

    @property
    def complete(self):
        return all(count == self.quota for count in self.counts)

    def step(self, dones, infos):
        if len(dones) != self.workers or len(infos) != self.workers:
            raise ValueError('Evaluation worker count changed')
        for worker, done in enumerate(dones):
            self.lengths[worker] += 1
            if not done:
                continue
            if self.counts[worker] < self.quota:
                info = infos[worker]
                slot = self.counts[worker]
                expected = self.seed_base + slot * self.stride + worker
                if int(info['eval_map_seed']) != expected or int(info['eval_map_slot']) != slot:
                    raise ValueError('Unexpected evaluation map order')
                if int(info['initial_geometry']['seed']) != expected:
                    raise ValueError('Terminal geometry belongs to a different episode')
                row = dict(worker=worker, episode_index_in_cycle=slot, eval_map_slot=slot,
                           eval_map_seed=expected, episode_length=self.lengths[worker])
                for name in ('success', 'crash', 'collision', 'timeout'):
                    row[name] = bool(info['is_' + name])
                for name in ('static_collision', 'dynamic_collision', 'dynamic_count',
                             'final_distance', 'min_distance', 'min_lidar_dist'):
                    row[name] = float(info[name])
                self.records.append(row)
                self.maps.append(info['initial_geometry'])
                self.counts[worker] += 1
            self.lengths[worker] = 0

    def finish(self):
        if not self.complete:
            raise RuntimeError(f'Incomplete evaluation: {self.counts}')
        records = sorted(self.records, key=lambda r: (r['worker'], r['eval_map_slot']))
        maps = sorted(self.maps, key=lambda r: r['seed'])
        if len({r['seed'] for r in maps}) != self.workers * self.quota:
            raise ValueError('Duplicate evaluation seed')
        return records, maps
